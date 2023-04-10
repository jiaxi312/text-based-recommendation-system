import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class Vectorize:
    """Text vectorization class to remove stopwords, punctuation and generate index dict for text data

    Attributes:
        words_to_remove: A set of words to be removed during vectorization (stop words, punctuations, etc.)
        vocabulary: A dict of mapping string token to index such like
                    {"hello": 1, "world": 2}
                    An empty string "" is placed to 0th place which can be used for padding.
                    A [UNK] indicates the token is not seen before and thus not indexed

        inverse_vocabulary: A dict of mapping index to string token, the inverse version of vocabulary such like
                    {1: "hello", 2: "world"}
    """

    def __init__(self):
        self.words_to_remove = set(stopwords.words('english')).union(string.punctuation)
        self.vocabulary = {"": 0, "[UNK]": 1}
        self.inverse_vocabulary = {0: "", "[UNK]": 1}

    def tokenize(self, text):
        """Tokenizes the input text data, including removing stop words and non-letter tokens"""
        text = text.lower()
        tokens = word_tokenize(text)
        return [token for token in tokens
                if token not in self.words_to_remove and self._is_valid_word(token)]

    def make_vocabulary(self, dataset):
        """Creates a vocabulary mapping based on the whole given dataset"""
        for text in dataset:
            self.update_vocabulary(text)

    def update_vocabulary(self, text):
        """Updates the token-index mapping based on the input text data."""
        tokens = self.tokenize(text)
        for token in tokens:
            if token not in self.vocabulary:
                # Only updates if the token hasn't been seen so far
                index = len(self)
                self.vocabulary[token] = index
                self.inverse_vocabulary[index] = token

    def encode(self, text, length=None):
        """Encodes the text data into given length based on the token-index mapping.

        Returns:
            A list of integers of the indicies corresponds to each token in the text"""
        tokens = self.tokenize(text)

        encoded = [self.vocabulary.get(token, 1) for token in tokens]
        if length is not None and len(encoded) < length:
            encoded.extend([0] * (length - len(encoded)))
        return encoded

    def decode(self, int_sequence):
        """Decodes the integer sequence back to string sequence."""
        return " ".join(
            self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)

    def __getitem__(self, item):
        if type(item) == int:
            return self.inverse_vocabulary.get(item, "[UNK]")
        return self.vocabulary.get(item, 1)

    def __len__(self):
        return len(self.vocabulary)

    @staticmethod
    def _is_valid_word(token):
        """Determines if a word is valid, meaning it contains at least one letter. """
        return re.search('[a-zA-Z]', token) is not None


def main():
    test_data = ["hello there", "nice to meet you", "there are some stopwords!!"]
    vectorize = Vectorize()
    vectorize.make_vocabulary(test_data)
    print(vectorize.encode(test_data[0], length=10))
    print(vectorize.decode(vectorize.encode(test_data[0], length=10)))


if __name__ == '__main__':
    main()