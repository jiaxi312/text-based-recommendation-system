import keras
import numpy as np

from keras import layers
from keras_resnet.models import ResNet1D34


class TextFeatureExtractorLayer(layers.Layer):
    """A deep neural network layer to extract features from text data using Resnet architecture.

    This layer uses the modified version of Resnet network for 1D input data. It uses the open-source network from
    keras-resnet.

    Source: https://github.com/broadinstitute/keras-resnet

    Attributes:
        input_dim: A tuple of integers representing the dimension of input data
        output_dim: An integer of the number of features for the final output
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resnet_layer = ResNet1D34(layers.Input(shape=input_dim), include_top=False)
        self.resnet_layer.layers.pop()
        self.fc_layer = keras.Sequential(
            [layers.Dropout(0.5),
             layers.Dense(units=1024, activation='relu'),
             layers.Dropout(0.5),
             layers.Dense(units=512, activation='relu'),
             layers.Dropout(0.5),
             layers.Dense(units=self.output_dim, activation='relu')
             ]
        )

    def call(self, inputs):
        inputs = self.resnet_layer(inputs)
        return self.fc_layer(inputs)


class GloveEmbeddingLayer(layers.Layer):
    """A pretrained embedding layer using Glove word matrix.

    An embedding layer loaded with the pretrained glove word matrix. The word matrix collects
    42 billion words and each word token has 300 dimensions.

    Source: https://nlp.stanford.edu/projects/glove/

    """

    def __init__(self, num_tokens, vocabulary_dict, path_to_glove_file='./glove/glove.42B.300d.txt',
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = 300
        self.num_tokens = num_tokens
        self.vocabulary_dict = vocabulary_dict

        self.embedding_matrix = self._load_embeddings_matrix(path_to_glove_file)

        self.embedding_layer = layers.Embedding(input_dim=self.num_tokens, output_dim=self.embed_dim,
                                                embeddings_initializer=keras.initializers.Constant(
                                                    self.embedding_matrix),
                                                trainable=False)

    def call(self, inputs):
        return self.embedding_layer(inputs)

    def _load_embeddings_matrix(self, path_to_glove_file):
        """Loads the token and its corresponding coefficients"""
        try:
            matrix = np.zeros((self.num_tokens, self.embed_dim))
            with open(path_to_glove_file, 'r') as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    if word in self.vocabulary_dict:
                        matrix[self.vocabulary_dict[word]] = coefs
            return matrix
        except FileNotFoundError:
            raise FileNotFoundError(
                f"\nNo glove file found at {path_to_glove_file} \n"
                f"Make sure you've downloaded the file at\n"
                f"https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors\n"
                f"And the path is correct")


def main():
    ...


if __name__ == '__main__':
    main()
