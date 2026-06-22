import csv
import numpy as np

class M4TransformerPreprocessor:
    def __init__(self, context_length=48, horizon=18):
        self.context_length = context_length
        self.horizon = horizon
        self.dataset = []
        self.val_stats = []

    def load_from_csv(self, file_path):
        self.dataset = []

        with open(file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
        
            for row in csv_reader:
                numeric_values = [float(v) for v in row[1:] if v]
                self.dataset.append(numeric_values)
        
        return self.dataset

    def _get_normalization_params(self, series):
        train_part = series[:-self.horizon]
        mean = train_part.mean()
        std = train_part.std()

        if std < 1e-8:
            std = 1.0
        
        return mean, std

    def split_dataset_by_series(self, val_ratio=0.2, dataset=None):
        if dataset is None:
            dataset = self.dataset

        eligible_series = [
            series for series in dataset
            if len(series) >= self.context_length + self.horizon + 1
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
            
            if len(series) < self.context_length + self.horizon + 1:
                continue

            mean, std = self._get_normalization_params(series)
            train_norm = (series[:-self.horizon] - mean) / std

            for start in range(0, len(train_norm) - self.context_length):
                chunk = train_norm[start : start + self.context_length + 1]
                X.append(chunk[:-1, None])
                y.append(chunk[1:, None])

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def get_validation_data(self, dataset=None):
        X_val, y_val = [], []
        self.val_stats = []

        if dataset is None:
            dataset = self.dataset

        for series in dataset:
            series = np.asarray(series, dtype=np.float32)

            if len(series) < self.context_length + self.horizon:
                continue

            mean, std = self._get_normalization_params(series)
            history = (series[-(self.context_length + self.horizon) : -self.horizon] - mean) / std
            future = series[-self.horizon:]

            X_val.append(history[:, None])
            y_val.append(future)
            self.val_stats.append((mean, std))

        return np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.float32)
    
    
    def load_from_csv_with_ids(self, file_path):
        series_dict = {}

        with open(file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)

            for row in csv_reader:
                series_id = row[0]
                numeric_values = [float(v) for v in row[1:] if v]
                series_dict[series_id] = numeric_values

        return series_dict


    def get_official_test_data(self, train_dict, test_dict):
        X_test, y_test = [], []
        self.val_stats = []

        skipped_short_history = 0
        skipped_missing_test = 0

        for series_id, history in train_dict.items():
            if series_id not in test_dict:
                skipped_missing_test += 1
                continue

            history = np.asarray(history, dtype=np.float32)
            future = np.asarray(test_dict[series_id], dtype=np.float32)

            if len(history) < self.context_length:
                skipped_short_history += 1
                continue

            if len(future) < self.horizon:
                continue

            mean = history.mean()
            std = history.std()

            if std < 1e-8:
                std = 1.0

            context = history[-self.context_length:]
            context_norm = (context - mean) / std

            X_test.append(context_norm[:, None])
            y_test.append(future[:self.horizon])
            self.val_stats.append((mean, std))

        print("Official test data")
        print("Used series:", len(X_test))
        print("Skipped short history:", skipped_short_history)
        print("Skipped missing test:", skipped_missing_test)

        return np.asarray(X_test, dtype=np.float32), np.asarray(y_test, dtype=np.float32)