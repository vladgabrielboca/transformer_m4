import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    CONTEXT_LENGTH, HORIZON, MAX_SERIES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, TRAIN_CSV_PATH, MODEL_SAVE_PATH,
    HIDDEN_DIM, INTERMEDIATE_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT_RATE
)

from src.preprocessing import M4TransformerPreprocessor
from src.model.transformer import build_decoder_only_transformer

preprocessor = M4TransformerPreprocessor(
    context_length=CONTEXT_LENGTH,
    horizon=HORIZON
)

dataset = preprocessor.load_from_csv(TRAIN_CSV_PATH)

if MAX_SERIES is not None:
    preprocessor.dataset = dataset[:MAX_SERIES]
else:
    preprocessor.dataset = dataset

X_train_full, y_train_full = preprocessor.get_training_data()

split_idx = int(len(X_train_full) * 0.8)

X_train = X_train_full[:split_idx]
y_train = y_train_full[:split_idx]
X_val_fit = X_train_full[split_idx:]
y_val_fit = y_train_full[split_idx:]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val_fit:", X_val_fit.shape)
print("y_val_fit:", y_val_fit.shape)

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=min(len(X_train), 10000), reshuffle_each_iteration=True)
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

checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)