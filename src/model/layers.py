import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.python.keras.layers import Dense

class DecoderBlock(layers.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        key_dim = hidden_dim // num_heads

        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate
        )
        
        self.attention_dropout = layers.Dropout(dropout_rate)
        self.attention_layernorm = layers.LayerNormalization(epsilon=1e-6)

        self.ffn_dense_1 = layers.Dense(intermediate_dim, activation="relu")
        self.ffn_dropout = layers.Dropout(dropout_rate)
        self.ffn_dense_2 = layers.Dense(hidden_dim)
        self.ffn_layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        # masked self-attention
        residual = x
        x = self.self_attention(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True,
            training=training
        )
        
        x = self.attention_dropout(x, training=training)
        x = x + residual
        x = self.attention_layernorm(x)

        # feed-forward
        residual = x
        x = self.ffn_dense_1(x)
        x = self.ffn_dropout(x, training=training)
        x = self.ffn_dense_2(x)
        x = x + residual
        x = self.ffn_layernorm(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config