import re
import string
import json

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
    def __init__(self, fpath='./data/filter_all_t.json', text_vectorize=None):
        self.fpath = fpath
        self.vectorize = text_vectorize
        all_raw_data = self.load_google_restaurants_dataset()

        (self.train_user_profiles,
         self.train_bus_profiles,
         self.train_records) = self._build_profiles_from(all_raw_data['train'], update_vectorize=True)

        (self.test_user_profiles,
         self.test_bus_profiles,
         self.test_records) = self._build_profiles_from(all_raw_data['test'])

        del all_raw_data

    def load_google_restaurants_dataset(self):
        """Loads the Google Restaurants Review dataset into memory.

        Returns:
            all_raw_data: A dict of all the reviews
        """
        try:
            with open(self.fpath, 'r') as f:
                all_raw_data = json.load(f)
                return all_raw_data
        except FileNotFoundError:
            raise FileNotFoundError(
                f"google restaurants dataset is not found at {self.fpath} \n")

    def _build_profiles_from(self, raw_data, update_vectorize=False):
        """Creates user profiles and business profiles from the raw data.

        Returns:
            user_profiles: A dict with key being user_id and value being a list of all reviews
                            from that user_id
            business_profiles: A dict with key being business_id and value being a list of all reviews
                            left for that restaurant
            records: A list of (user_id, business_id, rating) that keeps track of each record. It will
                    be used later to generate training and testing data.
        """
        user_profiles = {}
        business_profiles = {}
        records = []
        for review in raw_data:
            user_id = review['user_id']
            business_id = review['business_id']
            text = review['review_text']
            rating = review['rating']
            if user_id not in user_profiles:
                user_profiles[user_id] = []
            if business_id not in business_profiles:
                business_profiles[business_id] = []

            user_profiles[user_id].append(text)
            business_profiles[business_id].append(text)
            records.append((user_id, business_id, rating))

            if self.vectorize is not None and update_vectorize:
                self.vectorize.update_vocabulary(text)

        return user_profiles, business_profiles, records


def main():
    # test_data = ["hello there", "nice to meet you", "there are some stopwords!!"]
    # vectorize = Vectorize()
    # vectorize.make_vocabulary(test_data)
    # print(vectorize.encode(test_data[0], length=10))
    # print(vectorize.decode(vectorize.encode(test_data[0], length=10)))
    dataset = GoogleRestaurantsReviewDataset()
    print(dataset.train_user_profiles)


if __name__ == '__main__':
    main()
