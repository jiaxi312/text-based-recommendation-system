from utils import GoogleRestaurantsReviewDataset


def main():
    dataset = GoogleRestaurantsReviewDataset()
    text_vectorize = dataset.text_vectorize

    print('Load training data')
    train_X_user, train_X_bus, train_y = dataset.load_train_or_test_dataset(train=True)
    print(f'Total {train_y.shape} training data\n')

    print('Load test data')
    test_X_user, test_X_bus, test_y = dataset.load_train_or_test_dataset(train=False)
    print(f'Total {test_y.shape} test data\n')

    print('Build model')


if __name__ == '__main__':
    main()
