import csv
import tensorflow as tf
import numpy as np
import urllib
from tensorflow import keras
from keras.layers import Dense, LSTM, Lambda, Conv1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.losses import Huber


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


if __name__ == '__main__':
    sunspots = []
    time_step = []
    time_train = None
    time_valid = None
    x_train = None
    x_valid = None
    # 윈도우 사이즈
    window_size = 30
    # 배치 사이즈
    batch_size = 32
    # 셔플 사이즈
    shuffle_size = 1000

    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # 첫 줄은 header이므로 skip 합니다.
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

    series = np.array(sunspots)
    time = np.array(time_step)
    split_time = 3000
    time_train = time[:split_time]
    time_valid = time[split_time:]

    x_train = series[:split_time]
    x_valid = series[split_time:]

    train_set = windowed_dataset(x_train,
                                 window_size=window_size,
                                 batch_size=batch_size,
                                 shuffle_buffer=shuffle_size)

    validation_set = windowed_dataset(x_valid,
                                      window_size=window_size,
                                      batch_size=batch_size,
                                      shuffle_buffer=shuffle_size)

    model = Sequential([
        tf.keras.layers.Conv1D(60, kernel_size=5,
                               padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)  ########### 문제에 따라 제공안되었을 경우 제외
    ])

    optimizer = SGD(lr=1e-5, momentum=0.9)
    loss = Huber()
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=["mae"])

    checkpoint_path = 'tmp_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_mae',
                                 verbose=1)

    epochs = 100
    history = model.fit(train_set,
                        validation_data=(validation_set),
                        epochs=epochs,
                        callbacks=[checkpoint],
                        )
    model.load_weights(checkpoint_path)
    model.save("TF5-sunspot-02.h5")
    pass
