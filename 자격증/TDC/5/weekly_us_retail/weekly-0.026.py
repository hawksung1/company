import urllib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# DO NOT CHANGE THIS.
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    # DO NOT CHANGE THIS CODE
    # Reads the dataset.
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)

    N_FEATURES = len(df.columns)  # DO NOT CHANGE THIS

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.8)  # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS
    N_PAST = 10  # DO NOT CHANGE THIS
    N_FUTURE = 10  # DO NOT CHANGE THIS
    SHIFT = 1  # DO NOT CHANGE THIS

    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    # Code to define your model.
    model = tf.keras.models.Sequential([
        Conv1D(filters=32, kernel_size=5, padding='causal', activation='relu', input_shape=[N_PAST, 1]),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        tf.keras.layers.Dense(N_FEATURES)
    ])
    checkpoint_path = '05-weekly.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_mae',
                                 verbose=1)

    optimizer = tf.keras.optimizers.Adam(0.00003)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])

    model.fit(train_set, validation_data=(valid_set), epochs=150, callbacks=[checkpoint])

    model.load_weights(checkpoint_path)
    model.evaluate(valid_set)
    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF5-weekly-us-retail.h5")