import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing import M4TransformerPreprocessor
from src.model.embedding import PositionalEmbedding
from src.model.layers import DecoderBlock

preprocessor = M4TransformerPreprocessor(context_length=48, horizon=18)
dataset = preprocessor.load_from_csv('./data/Monthly-train.csv')

X_train, y_train = preprocessor.get_training_data()
X_val, y_val = preprocessor.get_validation_data()

def build_decoder_only_transformer(
    context_length=48,
    hidden_dim=64,
    intermediate_dim=128,
    num_heads=4,
    num_layers=3,
    dropout_rate=0.1
):
    inputs = keras.Input(shape=(context_length, 1))

    x = PositionalEmbedding(context_length, hidden_dim)(inputs)

    for _ in range(num_layers):
        x = DecoderBlock(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs, name="decoder_only_ts_transformer")