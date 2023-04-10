import keras
import numpy as np

from keras import layers


class GloveEmbedding(layers.Layer):
    """A pretrained embedding layer using Glove word matrix.

    An embedding layer loaded with the pretrained glove word matrix. The word matrix collects
    42 billion words and each word token has 300 dimensions.

    Source: https://nlp.stanford.edu/projects/glove/

    """

    def __init__(self, num_tokens, vocab_size, vocabulary_dict, path_to_glove_file='./glove/glove.42B.300d.txt',
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = 300
        self.num_tokens = num_tokens
        self.vocab_size = vocab_size
        self.vocabulary_dict = vocabulary_dict

        self.embedding_index = self._load_embeddings_index(path_to_glove_file)
        self.embedding_matrix = self._load_embeddings_matrix()
        del self.embedding_index

        self.embedding_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                                embeddings_initializer=keras.initializers.Constant(
                                                    self.embedding_matrix),
                                                trainable=False)

    def call(self, inputs):
        return self.embedding_layer(inputs)

    @staticmethod
    def _load_embeddings_index(path_to_glove_file):
        """Loads the token and its corresponding coefficients"""
        try:
            embeddings_index = {}
            with open(path_to_glove_file, 'r') as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    embeddings_index[word] = coefs
            return embeddings_index
        except FileNotFoundError:
            raise FileNotFoundError(
                f"\nNo glove file found at {path_to_glove_file} \n"
                f"Make sure you've downloaded the file at\n"
                f"https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors\n"
                f"And the path is correct")

    def _load_embeddings_matrix(self):
        """Convert the embedding vector into matrix based on the vocabulary dict"""
        hits = 0
        misses = 0
        embeddings_matrix = np.zeros((self.num_tokens, self.embedding_dim))

        for word, i in self.vocabulary_dict.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embeddings_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1

        print("Converted %d words (%d misses)" % (hits, misses))
        return embeddings_matrix


def main():
    ...


if __name__ == '__main__':
    main()
