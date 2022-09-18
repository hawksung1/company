import urllib
import os
import zipfile
import pandas as pd

import tensorflow as tf
from keras.layers import Dense, Conv1D, LSTM, Bidirectional
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


# 아래 2줄 코드는 넣지 말아 주세요!!!
url = 'https://www.dropbox.com/s/eduk281didil1km/Weekly_U.S.Diesel_Retail_Prices.csv?dl=1'
urllib.request.urlretrieve(url, 'Weekly_U.S.Diesel_Retail_Prices.csv')

df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                 infer_datetime_format=True, index_col='Week of', header=0)

N_FEATURES = len(df.columns)  # DO NOT CHANGE THIS

data = df.values
data = normalize_series(data, data.min(axis=0), data.max(axis=0))

SPLIT_TIME = int(len(data) * 0.8)  # DO NOT CHANGE THIS
x_train = data[:SPLIT_TIME]
x_valid = data[SPLIT_TIME:]

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

model = tf.keras.models.Sequential([
    Conv1D(filters=64, kernel_size=5, padding='causal', activation='relu', input_shape=[N_PAST, 1]),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),

    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(N_FEATURES),

    tf.keras.layers.Dense(N_FEATURES)
])


checkpoint_path = "tmp_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_mae',
                             verbose=1)
optimizer = tf.keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])
model.fit(train_set, validation_data=valid_set, epochs=100, callbacks=checkpoint)
model.load_weights(checkpoint_path)
model.evaluate(valid_set)
model.save("TF5-weekly-us-retail-0.027.h5")