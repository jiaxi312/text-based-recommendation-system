import tensorflow as tf

from tensorflow import keras
from keras import layers


class TransformerEncoder(layers.Layer):
    """A deep neural network using attention mechanism to encode the input sequence data.

    Attribute:
        embed_dim: An integer size of the input token vector
        dense_dim: An integer size of the inner dense layer
        num_heads: An integer number of attention heads
    """

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(self.dense_dim, activation='relu'),
             layers.Dense(self.embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config


class PositionalEmbedding(layers.Layer):
    """An embedding layer with token position being considered.

    Attributes:
        sequence_length: An integer length of the input sequence
        input_dim: An integer of the dimension of the input sequence
        output_dim: An integer size of expected output sequence
    """

    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


def main():
    # some test code for the model
    vocab_size = 10000
    sequence_length = 500
    embed_dim = 64
    num_heads = 2
    dense_dim = 32

    inputs = keras.Input(shape=(None,), dtype="int64")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="relu")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"])
    print(model.summary())


if __name__ == '__main__':
    main()
