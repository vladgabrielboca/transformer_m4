from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from src.preprocessing import M4TransformerPreprocessor
from src.model.transformer import build_decoder_only_transformer

preprocessor = M4TransformerPreprocessor(context_length=48, horizon=18)
dataset = preprocessor.load_from_csv('./data/Monthly-train.csv')
X_train, y_train = preprocessor.get_training_data()
model = build_decoder_only_transformer()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

checkpoint = ModelCheckpoint(
    filepath="transformer_best_weights.keras",
    monitor="loss",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=64,
    callbacks=[checkpoint]
)