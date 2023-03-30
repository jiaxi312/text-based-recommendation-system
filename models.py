import tensorflow as tf

from tensorflow import keras
from keras import layers


class TransformerEncoder(layers.Layer):
    """A deep neural network using attention mechanism to encode the input sequence data.

    The model is a simplified version of the transformer architecture mentioned in the original paper
    https://arxiv.org/pdf/1706.03762v5.pdf
    This model encodes the given sequence to a high-dimensional numeric data.

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

    The embedding layer is responsible to find the similarity between tokens,
    also this layer considers the position of each token in the original input
    sequence.

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


class DeepFM(layers.Layer):
    """A neural network to exam the interation between features and predict the preferrence

    This is a modified version of DeepFM model from original paper
    https://arxiv.org/abs/1703.04247
    In this model, the embedding layers are removed, since the features are all
    dense features generated from previous neural network. And embedding is
    used in previous layers.
    The model is also modified to do regression task instead of binary classification,
    though it can be changed to do classification problem if needed

    Attribute:
        num_feat: An integer of the number of features
        hidden_units: An integer list or iterable of the size of hidden dense layer
        output_dim: An integer of the final output dimension
    """

    def __init__(self, num_feat, hidden_units=(256, 128, 64), output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.num_feat = num_feat
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        self.dnn = self.create_dnn()
        self.prediction = layers.Dense(units=output_dim, activation='relu')

    def call(self, inputs):
        fm_outputs = self.fm(inputs)
        dnn_outputs = self.dnn(inputs)
        return self.prediction(tf.concat([fm_outputs, dnn_outputs], 1))

    def fm(self, fm_input):
        """FM factorization of the input feature.

        The factorization exmas both linear (order-1) and pairwise (order-2)
        interation. The linear interaction is computed using addition, and
        pairwise interaction with inner product.
        """
        sum_square = tf.square(tf.reduce_sum(fm_input, axis=1, keepdims=True))
        square_sum = tf.reduce_sum(tf.square(fm_input), axis=1, keepdims=True)
        fm_output = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), axis=1, keepdims=True)
        return fm_output

    def create_dnn(self):
        """Creates a sequential fully connected layers based on the hidden units"""
        return keras.Sequential(
            [layers.Dense(units=unit, activation='relu')
             for unit in self.hidden_units]
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_feat": self.num_feat,
            "hidden_units": self.hidden_units,
            "output_dim": self.output_dim
        })
        return config


def main():
    # some test code for the model
    vocab_size = 30000
    sequence_length = 500
    embed_dim = 64
    num_heads = 2
    dense_dim = 32
    num_feat = 64

    # the basic transformer encoder model, the input data will first go through
    # the position embedding layer which can find the similarity and positional
    # information of each token. Then, goes through the transformer layer, which
    # we hope to extract some useful information. Finally (not implemented),
    # those information will go through DeepFM model
    pos_embed = PositionalEmbedding(sequence_length, vocab_size, embed_dim)
    user_profile_inputs = keras.Input(shape=(None,), dtype="int64")
    x = pos_embed(user_profile_inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    user_profile_outputs = layers.Dense(num_feat, activation="relu")(x)

    bus_profile_inputs = keras.Input(shape=(None,), dtype="int64")
    x = pos_embed(bus_profile_inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    bus_profile_outputs = layers.Dense(num_feat, activation="relu")(x)

    outputs = DeepFM(num_feat * 2)(
        layers.concatenate([user_profile_outputs, bus_profile_outputs], axis=1))

    model = keras.Model([user_profile_inputs, bus_profile_inputs], outputs)
    model.compile(optimizer="rmsprop",
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"])
    print(model.summary())


if __name__ == '__main__':
    main()
