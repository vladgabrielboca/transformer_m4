import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

from experiments.logratio.config import (
    CONTEXT_LENGTH, HORIZON, MAX_SERIES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, TRAIN_CSV_PATH, MODEL_SAVE_PATH,
    HIDDEN_DIM, INTERMEDIATE_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT_RATE,
    SERIES_VAL_RATIO, RANDOM_SEED, PREDICT_BATCH_SIZE, AUTOREGRESSIVE_PATIENCE,
    LOG_RATIO_CLIP, LOG_EPSILON
)

from experiments.logratio.preprocessing import M4LogRatioPreprocessor
from src.model.transformer import build_decoder_only_transformer


def compute_smape(y_true, y_pred):
    eps = 1e-8
    return 200 * np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )


def autoregressive_logratio_forecast_batch(
    model,
    X_val,
    forecast_anchors,
    horizon=18,
    batch_size=512,
    log_ratio_clip=5.0
):
    window = X_val.astype(np.float32).copy()
    current_values = np.asarray(forecast_anchors, dtype=np.float32)
    preds_raw = []

    for _ in range(horizon):
        pred_seq = model.predict(window, batch_size=batch_size, verbose=0)
        next_log_ratios = pred_seq[:, -1, 0]

        if log_ratio_clip is not None:
            next_log_ratios = np.clip(next_log_ratios, -log_ratio_clip, log_ratio_clip)

        current_values = current_values * np.exp(next_log_ratios)
        preds_raw.append(current_values.copy())

        next_values = next_log_ratios[:, None, None].astype(np.float32)
        window = np.concatenate([window[:, 1:, :], next_values], axis=1)

    return np.stack(preds_raw, axis=1).astype(np.float32)


class AutoregressiveLogRatioValidationCallback(Callback):
    def __init__(self, X_val, y_val, forecast_anchors, horizon, batch_size, save_path, patience, log_ratio_clip):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.forecast_anchors = forecast_anchors
        self.horizon = horizon
        self.batch_size = batch_size
        self.save_path = save_path
        self.patience = patience
        self.log_ratio_clip = log_ratio_clip
        self.best_smape = np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        preds = autoregressive_logratio_forecast_batch(
            model=self.model,
            X_val=self.X_val,
            forecast_anchors=self.forecast_anchors,
            horizon=self.horizon,
            batch_size=self.batch_size,
            log_ratio_clip=self.log_ratio_clip
        )
        val_smape_ar = compute_smape(self.y_val, preds)
        logs["val_smape_ar"] = val_smape_ar

        print(f"\nEpoch {epoch + 1}: val_smape_ar = {val_smape_ar:.4f}")

        if val_smape_ar < self.best_smape:
            self.best_smape = val_smape_ar
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.model.save(self.save_path)
            print(f"Epoch {epoch + 1}: model saved in {self.save_path} (best val_smape_ar)")
        else:
            self.wait += 1
            print(f"Epoch {epoch + 1}: no improvement on val_smape_ar ({self.wait}/{self.patience})")
            if self.wait >= self.patience:
                self.model.stop_training = True
                print("Early stopping activated based on val_smape_ar.")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

preprocessor = M4LogRatioPreprocessor(
    context_length=CONTEXT_LENGTH,
    horizon=HORIZON,
    epsilon=LOG_EPSILON,
    log_ratio_clip=LOG_RATIO_CLIP
)

dataset = preprocessor.load_from_csv(TRAIN_CSV_PATH)

if MAX_SERIES is not None:
    preprocessor.dataset = dataset[:MAX_SERIES]
else:
    preprocessor.dataset = dataset

train_series, val_series = preprocessor.split_dataset_by_series(
    val_ratio=SERIES_VAL_RATIO,
    dataset=preprocessor.dataset
)

print("Experiment: log-ratio regression")
print("Model context length:", CONTEXT_LENGTH)
print("Model save path:", MODEL_SAVE_PATH)

X_train, y_train = preprocessor.get_training_data(dataset=train_series)
X_val_fit, y_val_fit = preprocessor.get_training_data(dataset=val_series)
X_val_ar, y_val_ar = preprocessor.get_validation_data(dataset=val_series)
forecast_anchors = list(preprocessor.forecast_anchors)

print("Number of train series:", len(train_series))
print("Number of validation series:", len(val_series))
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val_fit:", X_val_fit.shape)
print("y_val_fit:", y_val_fit.shape)
print("X_val_ar:", X_val_ar.shape)
print("y_val_ar:", y_val_ar.shape)

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=min(len(X_train), 100000), reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val_fit, y_val_fit))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

model = build_decoder_only_transformer(
    context_length=CONTEXT_LENGTH,
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout_rate=DROPOUT_RATE
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

model.summary()

ar_val_callback = AutoregressiveLogRatioValidationCallback(
    X_val=X_val_ar,
    y_val=y_val_ar,
    forecast_anchors=forecast_anchors,
    horizon=HORIZON,
    batch_size=PREDICT_BATCH_SIZE,
    save_path=MODEL_SAVE_PATH,
    patience=AUTOREGRESSIVE_PATIENCE,
    log_ratio_clip=LOG_RATIO_CLIP
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ar_val_callback]
)
