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

    def split_dataset_by_series(self, val_ratio=0.2, seed=42, dataset=None):
        if dataset is None:
            dataset = self.dataset

        eligible_series = [
            series for series in dataset
            if len(series) >= self.context_length + self.horizon + 1
        ]

        if len(eligible_series) < 2:
            raise ValueError("At least 2 series needed for train/validation split.")

        rng = np.random.default_rng(seed)
        indices = np.arange(len(eligible_series))
        rng.shuffle(indices)

        val_size = max(1, int(len(eligible_series) * val_ratio))
        if val_size >= len(eligible_series):
            val_size = len(eligible_series) - 1

        val_indices = set(indices[:val_size].tolist())

        train_series = [eligible_series[i] for i in range(len(eligible_series)) if i not in val_indices]
        val_series = [eligible_series[i] for i in range(len(eligible_series)) if i in val_indices]

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