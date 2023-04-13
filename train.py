import numpy as np
from tqdm import tqdm

from utils import Vectorize, GoogleRestaurantsReviewDataset


def build_data_from_profiles(dataset, text_vectorize, sequence_length, train=True):
    user_profiles = dataset.train_user_profiles if train else dataset.test_user_profiles
    bus_profiles = dataset.train_bus_profiles if train else dataset.test_bus_profiles
    records = dataset.train_records if train else dataset.test_records

    num_record = len(records)
    X_user = np.zeros((num_record, sequence_length))
    X_bus = np.zeros((num_record, sequence_length))
    y = np.zeros(num_record)

    for i in tqdm(range(num_record)):
        user_id, business_id, rating = records[i]
        user_review = ' '.join(user_profiles[user_id])
        bus_review = ' '.join(bus_profiles[business_id])

        user_review_vector = text_vectorize.encode(user_review, sequence_length)
        bus_review_vector = text_vectorize.encode(bus_review, sequence_length)

        X_user[i] = np.array(user_review_vector)
        X_bus[i] = np.array(bus_review_vector)
        y[i] = rating

    return X_user, X_bus, y


def main():
    text_vectorize = Vectorize()
    dataset = GoogleRestaurantsReviewDataset(text_vectorize=text_vectorize)

    print('Load training data')
    train_X_user, train_X_bus, train_y = build_data_from_profiles(dataset,
                                                                  text_vectorize,
                                                                  sequence_length=500)
    print(f'Total {train_y.shape} training data\n')

    print('Load test data')
    test_X_user, test_X_bus, test_y = build_data_from_profiles(dataset,
                                                               text_vectorize,
                                                               sequence_length=500,
                                                               train=False)
    print(f'Total {test_y.shape} test data\n')

    print('Build model')

    del dataset


if __name__ == '__main__':
    main()
