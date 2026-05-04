import tensorflow as tf
from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    def __init__(self, context_length, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.context_length = context_length
        self.hidden_dim = hidden_dim

        self.value_projection = layers.Dense(hidden_dim)

        self.position_embedding = layers.Embedding(
            input_dim=context_length,
            output_dim=hidden_dim
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embed = self.position_embedding(positions)
        pos_embed = tf.expand_dims(pos_embed, axis=0)
        pos_embed = tf.tile(pos_embed, [batch_size, 1, 1])

        value_embed = self.value_projection(x)
        return value_embed + pos_embed
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "context_length": self.context_length,
            "hidden_dim": self.hidden_dim,
        })
        return config