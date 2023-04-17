import tensorflow as tf

import keras
from keras.losses import cosine_similarity

from models import TextFeatureExtractorLayer, GloveEmbeddingLayer
from utils import GoogleRestaurantsReviewDataset


def main():
    max_seq_length = 750
    dataset = GoogleRestaurantsReviewDataset(max_seq_length=max_seq_length)
    text_vectorize = dataset.text_vectorize

    print('Load training data')
    train_X_user, train_X_bus, train_y = dataset.load_train_or_test_dataset(train=True)
    print(f'Total {train_y.shape} training data\n')

    print('Load test data')
    test_X_user, test_X_bus, test_y = dataset.load_train_or_test_dataset(train=False)
    print(f'Total {test_y.shape} test data\n')

    print('Build model')
    embedding = GloveEmbeddingLayer(num_tokens=len(text_vectorize),
                                    vocabulary_dict=text_vectorize.vocabulary)

    user_inputs = keras.Input(shape=(None,), dtype="int64")
    x = embedding(user_inputs)
    user_outputs = TextFeatureExtractorLayer(
        input_dim=(dataset.max_seq_length, embedding.embed_dim), output_dim=32)(x)

    bus_inputs = keras.Input(shape=(None,), dtype="int64")
    y = embedding(bus_inputs)
    bus_outputs = TextFeatureExtractorLayer(
        input_dim=(dataset.max_seq_length, embedding.embed_dim), output_dim=32)(y)

    outputs = -cosine_similarity(user_outputs, bus_outputs, axis=1)

    model = keras.Model([user_inputs, bus_inputs], outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    model.summary()

    print('Train the model')
    checkpoint_path = "./training_1/cp.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    # model.load_weights(checkpoint_path)
    model.fit([train_X_user, train_X_bus],
              train_y,
              epochs=8,
              validation_data=([test_X_user, test_X_bus], test_y),
              callbacks=[cp_callback, tensorboard_callback])


if __name__ == '__main__':
    main()
