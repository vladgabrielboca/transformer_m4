from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.preprocessing import M4TransformerPreprocessor
from src.model.transformer import build_decoder_only_transformer

preprocessor = M4TransformerPreprocessor(context_length=48, horizon=18)
dataset = preprocessor.load_from_csv('./data/Monthly-train.csv')

preprocessor.dataset = dataset[:5000] # LIMITA PENTRU COLAB

X_train_full, y_train_full = preprocessor.get_training_data()

split_idx = int(len(X_train_full) * 0.8)

X_train = X_train_full[:split_idx]
y_train = y_train_full[:split_idx]
X_val_fit = X_train_full[split_idx:]
y_val_fit = y_train_full[split_idx:]

model = build_decoder_only_transformer()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

checkpoint = ModelCheckpoint(
    filepath="transformer_best_weights.keras",
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
    X_train, y_train,
    validation_data=(X_val_fit, y_val_fit),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stop]
)