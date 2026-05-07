import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    CONTEXT_LENGTH, HIDDEN_DIM, INTERMEDIATE_DIM, 
    NUM_HEADS, NUM_LAYERS, DROPOUT_RATE
)

from src.preprocessing import M4TransformerPreprocessor
from src.model.embedding import PositionalEmbedding
from src.model.layers import DecoderBlock

def build_decoder_only_transformer(
    context_length=CONTEXT_LENGTH,
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout_rate=DROPOUT_RATE
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