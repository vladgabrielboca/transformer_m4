import csv
import numpy as np


class M4LogRatioPreprocessor:
    """
    Preprocessor for log-ratio regression.

    The model receives sequences of log-ratios:
        r[t] = log(x[t] / x[t-1])

    Forecasts are reconstructed autoregressively with:
        x_hat[t+1] = x_hat[t] * exp(r_hat[t+1])
    """

    def __init__(self, context_length=60, horizon=18, epsilon=1e-6, log_ratio_clip=5.0):
        self.context_length = context_length
        self.horizon = horizon
        self.epsilon = epsilon
        self.log_ratio_clip = log_ratio_clip
        self.dataset = []
        self.forecast_anchors = []

    def load_from_csv(self, file_path):
        self.dataset = []

        with open(file_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)

            for row in csv_reader:
                numeric_values = [float(v) for v in row[1:] if v]
                self.dataset.append(numeric_values)

        return self.dataset

    def load_from_csv_with_ids(self, file_path):
        series_dict = {}

        with open(file_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)

            for row in csv_reader:
                series_id = row[0]
                numeric_values = [float(v) for v in row[1:] if v]
                series_dict[series_id] = numeric_values

        return series_dict

    def _safe_log_ratios(self, values):
        values = np.asarray(values, dtype=np.float32)
        safe_values = np.maximum(values, self.epsilon)
        ratios = np.log(safe_values[1:] / safe_values[:-1])

        if self.log_ratio_clip is not None:
            ratios = np.clip(ratios, -self.log_ratio_clip, self.log_ratio_clip)

        return ratios.astype(np.float32)

    def split_dataset_by_series(self, val_ratio=0.2, seed=42, dataset=None):
        if dataset is None:
            dataset = self.dataset

        eligible_series = [
            series for series in dataset
            if len(series) >= self.context_length + self.horizon + 2
        ]

        if len(eligible_series) < 2:
            raise ValueError("At least 2 series needed for train/validation split.")

        val_size = max(1, int(len(eligible_series) * val_ratio))

        if val_size >= len(eligible_series):
            val_size = len(eligible_series) - 1

        split_idx = len(eligible_series) - val_size
        train_series = eligible_series[:split_idx]
        val_series = eligible_series[split_idx:]

        return train_series, val_series

    def get_training_data(self, dataset=None):
        X, y = [], []

        if dataset is None:
            dataset = self.dataset

        for series in dataset:
            series = np.asarray(series, dtype=np.float32)

            if len(series) < self.context_length + self.horizon + 2:
                continue

            train_values = series[:-self.horizon]
            train_log_ratios = self._safe_log_ratios(train_values)

            for start in range(0, len(train_log_ratios) - self.context_length):
                chunk = train_log_ratios[start : start + self.context_length + 1]
                X.append(chunk[:-1, None])
                y.append(chunk[1:, None])

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def get_validation_data(self, dataset=None):
        X_val, y_val = [], []
        self.forecast_anchors = []

        if dataset is None:
            dataset = self.dataset

        for series in dataset:
            series = np.asarray(series, dtype=np.float32)

            if len(series) < self.context_length + self.horizon + 1:
                continue

            history = series[-(self.context_length + self.horizon + 1) : -self.horizon]
            future = series[-self.horizon:]

            context_log_ratios = self._safe_log_ratios(history)

            if len(context_log_ratios) != self.context_length:
                continue

            X_val.append(context_log_ratios[:, None])
            y_val.append(future)
            self.forecast_anchors.append(float(history[-1]))

        return np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.float32)

    def get_official_test_data(self, train_dict, test_dict):
        X_test, y_test = [], []
        self.forecast_anchors = []

        skipped_short_history = 0
        skipped_missing_test = 0

        for series_id, history in train_dict.items():
            if series_id not in test_dict:
                skipped_missing_test += 1
                continue

            history = np.asarray(history, dtype=np.float32)
            future = np.asarray(test_dict[series_id], dtype=np.float32)

            if len(history) < self.context_length + 1:
                skipped_short_history += 1
                continue

            if len(future) < self.horizon:
                continue

            context = history[-(self.context_length + 1):]
            context_log_ratios = self._safe_log_ratios(context)

            if len(context_log_ratios) != self.context_length:
                continue

            X_test.append(context_log_ratios[:, None])
            y_test.append(future[:self.horizon])
            self.forecast_anchors.append(float(context[-1]))

        print("Official test data")
        print("Used series:", len(X_test))
        print("Skipped short history:", skipped_short_history)
        print("Skipped missing test:", skipped_missing_test)

        return np.asarray(X_test, dtype=np.float32), np.asarray(y_test, dtype=np.float32)
