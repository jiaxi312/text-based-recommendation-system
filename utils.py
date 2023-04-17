import csv
import json
import re
import string
import numpy as np

from tqdm import tqdm
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
        self.inverse_vocabulary = {0: "", 1: "[UNK]"}

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

        Args:
            text: A string of the text data needed to be encoded
            length: A integer of the maximum length of the encoded sequence.
                    The sequence larger than the maximum length will cut of and
                    the sequence smaller than the maximum length will be padded
                    with 0.

        Returns:
            A list of integers of the indicies corresponds to each token in the text
        """
        tokens = self.tokenize(text)
        if length is None:
            length = len(tokens)
        encoded = []
        for i in range(length):
            if i < len(tokens):
                encoded.append(self.vocabulary.get(tokens[i], 1))
            else:
                encoded.append(0)
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


class GoogleRestaurantsReviewDataset:
    """A class provides a convient way to load Google Restaurants Review Dataset

    Attributes:
        fpath: A string of the file path contains this dataset
        vectorize: A Vectorize object used to encode and decode text data.
    """

    def __init__(self, max_seq_length=500):
        self.text_vectorize = Vectorize()
        self.max_seq_length = max_seq_length

        print('Build Training data')
        self._build_training_data()
        print('Build Testing data')
        self._build_testing_data()

    def load_train_or_test_dataset(self, train=True):
        """Loads the training or testing dataset that can be used to the neural net.

        Returns:
            X_user: A 2D numpy matrix with each row being an encoded vector of all reviews from
                    that user in a record
            X_bus: A 2D numpy matrix with each row being an encoded vector of all reviews for
                    that restaurant in a record
            y: A 1D numpy array with the rating for each record. The ratings will be converted to 1
                if the original rating is 5, -1 otherwise.
        """
        user_profiles = self.train_user_reviews if train else self.test_user_reviews
        bus_profiles = self.train_bus_reviews if train else self.test_bus_reviews
        records = self.train_ratings if train else self.test_ratings

        return np.stack(user_profiles), np.stack(bus_profiles), np.stack(records)

    def _build_training_data(self):
        self.train_user_reviews = []
        self.train_bus_reviews = []
        self.train_ratings = []

        with open('./data/processed_training_data.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in tqdm(csv_reader):
                user_reviews = row[-2]
                bus_reviews = row[-1]
                rating = 1 if float(row[3]) == 5.0 else -1

                self.text_vectorize.update_vocabulary(user_reviews)
                self.text_vectorize.update_vocabulary(bus_reviews)

                self.train_user_reviews.append(
                    np.array(self.text_vectorize.encode(user_reviews, length=self.max_seq_length)))
                self.train_bus_reviews.append(
                    np.array(self.text_vectorize.encode(bus_reviews, length=self.max_seq_length)))
                self.train_ratings.append(rating)

    def _build_testing_data(self):
        self.test_user_reviews = []
        self.test_bus_reviews = []
        self.test_ratings = []

        with open('./data/processed_testing_data.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in tqdm(csv_reader):
                user_reviews = row[-2]
                bus_reviews = row[-1]
                rating = 1 if float(row[3]) == 5.0 else -1

                self.test_user_reviews.append(
                    np.array(self.text_vectorize.encode(user_reviews, length=self.max_seq_length)))
                self.test_bus_reviews.append(
                    np.array(self.text_vectorize.encode(bus_reviews, length=self.max_seq_length)))
                self.test_ratings.append(rating)


def main():
    # test_data = ["hello there", "nice to meet you", "there are some stopwords!!"]
    # vectorize = Vectorize()
    # vectorize.make_vocabulary(test_data)
    # print(vectorize.encode(test_data[0], length=10))
    # print(vectorize.decode(vectorize.encode(test_data[0], length=10)))
    dataset = GoogleRestaurantsReviewDataset()


if __name__ == '__main__':
    main()
