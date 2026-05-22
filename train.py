import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

from config import (
    CONTEXT_LENGTH, HORIZON, MAX_SERIES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, TRAIN_CSV_PATH, MODEL_SAVE_PATH,
    HIDDEN_DIM, INTERMEDIATE_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT_RATE,
    SERIES_VAL_RATIO, RANDOM_SEED, PREDICT_BATCH_SIZE, AUTOREGRESSIVE_PATIENCE
)

from src.preprocessing import M4TransformerPreprocessor
from src.model.transformer import build_decoder_only_transformer

def compute_smape(y_true, y_pred):
    eps = 1e-8
    return 200 * np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )

def autoregressive_forecast_batch(model, X_val, val_stats, horizon=18, batch_size=512):
    window = X_val.astype(np.float32).copy()
    preds_norm = []

    for _ in range(horizon):
        pred_seq = model.predict(window, batch_size=batch_size, verbose=0)
        next_values = pred_seq[:, -1, 0]
        preds_norm.append(next_values)
        next_values = next_values[:, None, None]
        window = np.concatenate([window[:, 1:, :], next_values], axis=1)

    preds_norm = np.stack(preds_norm, axis=1)
    means = np.asarray([mean for mean, _ in val_stats], dtype=np.float32)[:, None]
    stds = np.asarray([std for _, std in val_stats], dtype=np.float32)[:, None]
    preds_raw = preds_norm * stds + means

    return preds_raw

class AutoregressiveValidationCallback(Callback):
    def __init__(self, X_val, y_val, val_stats, horizon, batch_size, save_path, patience):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.val_stats = val_stats
        self.horizon = horizon
        self.batch_size = batch_size
        self.save_path = save_path
        self.patience = patience
        self.best_smape = np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        preds = autoregressive_forecast_batch(
            model=self.model,
            X_val=self.X_val,
            val_stats=self.val_stats,
            horizon=self.horizon,
            batch_size=self.batch_size
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

preprocessor = M4TransformerPreprocessor(
    context_length=CONTEXT_LENGTH,
    horizon=HORIZON
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

print("Model context length:", CONTEXT_LENGTH)

X_train, y_train = preprocessor.get_training_data(dataset=train_series)
X_val_fit, y_val_fit = preprocessor.get_training_data(dataset=val_series)
X_val_ar, y_val_ar = preprocessor.get_validation_data(dataset=val_series)
val_stats = list(preprocessor.val_stats)

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

ar_val_callback = AutoregressiveValidationCallback(
    X_val=X_val_ar,
    y_val=y_val_ar,
    val_stats=val_stats,
    horizon=HORIZON,
    batch_size=PREDICT_BATCH_SIZE,
    save_path=MODEL_SAVE_PATH,
    patience=AUTOREGRESSIVE_PATIENCE
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ar_val_callback]
)
