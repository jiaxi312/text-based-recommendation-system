import re
import string
import json
import numpy as np

from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords


class Vectorize:
    """Text vectorization class to remove stopwords, punctuation and generate index dict for text data"""
    def __init__(self):
        self.words_to_remove = set(stopwords.words("english")).union(string.punctuation)
        self.vocabulary = {"": 0, "[UNK]": 1}
        self.inverse_vocabulary = {0: "", 1: "[UNK]"}

    def tokenize(self, text):
        text = text.lower()
        tokens = wordpunct_tokenize(text)
        return [token for token in tokens if token not in self.words_to_remove and self.is_valid(token)]

    def make_vocabulary(self, dataset):
        for text in dataset:
            self.update_vocabulary(text)

    def update_vocabulary(self, text):
        tokens = self.tokenize(text)
        for token in tokens:
            if token not in self.vocabulary:
                index = len(self.vocabulary)
                self.vocabulary[token] = index
                self.inverse_vocabulary[index] = token

    def encode(self, text, length=None):
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
        return " ".join(
            self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)

    def get_vocabulary(self):
        return [word for word in self.vocabulary]

    @staticmethod
    def is_valid(token):
        return re.search('[a-zA-Z]', token) is not None

    def __getitem__(self, item):
        if type(item) == int:
            return self.inverse_vocabulary.get(item, "[UNK]")
        return self.vocabulary.get(item, 1)

    def __len__(self):
        return len(self.vocabulary)


class GoogleRestaurantsDataset:
    def __init__(self, fpath, train=True, text_vectorize=None):
        self.fpath = fpath
        self.train = train
        self.text_vectorize = Vectorize() if text_vectorize is None else text_vectorize
        self.max_seq_len = 1000

        self.raw_data = self._load_data()
        self._build_profiles(self.raw_data)

    def get_all_reviews(self):
        user_reviews, business_reviews, ratings = [], [], []
        for (user_id, business_id, rating) in self.data:
            user_review = np.array(self.text_vectorize.encode(self.user_profiles[user_id], self.max_seq_len))
            user_reviews.append(user_review)
            business_review = np.array(self.text_vectorize.encode(self.business_profiles[business_id], self.max_seq_len))
            business_reviews.append(business_review)
            ratings.append(rating)
        return np.stack(user_reviews), np.stack(business_reviews), np.stack(ratings)

    def __len__(self):
        return len(self.data)


    def _load_data(self):
        with open(self.fpath, 'r') as f:
            all_data = json.load(f)
            if self.train:
                return all_data['train']
            return all_data['test']

    def _build_profiles(self, raw_data):
        self.data = []
        self.user_profiles = {}
        self.business_profiles = {}
        for review in raw_data:
            user_id = review['user_id']
            business_id = review['business_id']
            text = review['review_text']
            rating = review['rating']

            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = []
            if business_id not in self.business_profiles:
                self.business_profiles[business_id] = []

            self.user_profiles[user_id].append(text)
            self.business_profiles[business_id].append(text)

            self.data.append((user_id, business_id, rating))
            self.text_vectorize.update_vocabulary(text)

        for user_id in self.user_profiles:
            self.user_profiles[user_id] = "".join(self.user_profiles[user_id])
        for business_id in self.business_profiles:
            self.business_profiles[business_id] = "".join(self.business_profiles[business_id])


def main():
    google_dataset = GoogleRestaurantsDataset(train=True)
    # print(google_dataset.get_all_user_reviews().shape)
    user_reviews, business_reviews, ratings = google_dataset.get_all_user_reviews()
    print(user_reviews.shape, business_reviews.shape, ratings.shape)


if __name__ == '__main__':
    main()
