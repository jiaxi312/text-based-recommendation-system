import tensorflow as tf

from tensorflow import keras
from keras import layers

from utils import GoogleRestaurantsDataset
from models import TransformerEncoder, PositionalEmbedding, DeepFM


def main():
    dataset = GoogleRestaurantsDataset(fpath="/Users/jiaxi312/Desktop/INF385T/Final_Project/filter_all_t.json",
                                       train=True)
    user_reviews, business_reviews, ratings = dataset.get_all_reviews()

    vocab_size = len(dataset.text_vectorize)
    sequence_length = 1000
    embed_dim = 64
    num_heads = 8
    dense_dim = 2048
    num_feat = 128

    dnn_layers = keras.Sequential(
        [PositionalEmbedding(sequence_length, vocab_size, embed_dim),
         TransformerEncoder(embed_dim, dense_dim, num_heads),
         layers.GlobalMaxPool1D(),
         layers.Dropout(0.2),
         layers.Dense(num_feat // 2, activation='relu')]
    )

    user_inputs = keras.Input(shape=(None,), dtype="int64")
    user_outputs = dnn_layers(user_inputs)
    business_inputs = keras.Input(shape=(None,), dtype="int64")
    business_outputs = dnn_layers(business_inputs)
    outputs = DeepFM(num_feat)(layers.concatenate([user_outputs, business_outputs], axis=1))

    model = keras.Model([user_inputs, business_inputs], outputs)

    model.compile(optimizer="rmsprop",
                  loss="mse",
                  metrics=["mean_squared_error"])
    model.summary()

    checkpoint_path = "training_1/cp.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit([user_reviews, business_reviews],
              ratings,
              epochs=10,
              callbacks=[cp_callback])


if __name__ == '__main__':
    main()
