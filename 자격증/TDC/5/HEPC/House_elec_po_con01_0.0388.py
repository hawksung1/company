# ==============================================================================
#
# TIME SERIES QUESTION
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model
# should be trained to predict the next 24 observations of the 7 variables.
#
# ==============================================================================

# =========== 합격 기준 가이드라인 공유 ============= #
# 2021년 7월 1일 신규 출제된 문제                     #
# 5/5가 잘 나오지 않으므로 모델 많이 만들어 둘 것     #
# =================================================== #
# 문제명: Category 5 - household electric power consumption
# val_loss: 0.053
# val_mae: 0.053 (val_loss와 동일)
# =================================================== #
# =================================================== #

# ABOUT THE DATASET
#
# Original Source:
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
#
# The original 'Individual House Hold Electric Power Consumption Dataset'
# has Measurements of electric power consumption in one household with
# a one-minute sampling rate over a period of almost 4 years.
#
# Different electrical quantities and some sub-metering values are available.
#
# For the purpose of the examination we have provided a subset containing
# the data for the first 60 days in the dataset. We have also cleaned the
# dataset beforehand to remove missing values. The dataset is provided as a
# csv file in the project.
#
# The dataset has a total of 7 features ordered by time.
# ==============================================================================
#
# INSTRUCTIONS
#
# Complete the code in following functions:
# 1. windowed_dataset()
# 2. solution_model()
#
# The model input and output shapes must match the following
# specifications.
#
# 1. Model input_shape must be (BATCH_SIZE, N_PAST = 24, N_FEATURES = 7),
#    since the testing infrastructure expects a window of past N_PAST = 24
#    observations of the 7 features to predict the next 24 observations of
#    the same features.
#
# 2. Model output_shape must be (BATCH_SIZE, N_FUTURE = 24, N_FEATURES = 7)
#
# 3. DON'T change the values of the following constants
#    N_PAST, N_FUTURE, SHIFT in the windowed_dataset()
#    BATCH_SIZE in solution_model() (See code for additional note on
#    BATCH_SIZE).
# 4. Code for normalizing the data is provided - DON't change it.
#    Changing the normalizing code will affect your score.
#
# HINT: Your neural network must have a validation MAE of approximately 0.055 or
# less on the normalized validation dataset for top marks.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.
#
# WARNING: If you are using the GRU layer, it is advised not to use the
# 'recurrent_dropout' argument (you can alternatively set it to 0),
# since it has not been implemented in the cuDNN kernel and may
# result in much longer training times.
import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Conv1D, LSTM, Bidirectional
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


# This function downloads and extracts the dataset to the directory that
# contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change https to http)
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=(n_past + n_future), shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.shuffle(len(series))
    ds = ds.map(
        lambda w: (w[:n_past], w[n_past:])
    )
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    download_and_extract_data()
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)
    N_FEATURES = len(df.columns)
    # Normalizes the data
    data = df.values
    split_time = int(len(data) * 0.5)
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))
    # Splits the data into training and validation sets.
    x_train = data[:split_time]
    x_valid = data[split_time:]
    # DO NOT CHANGE 'BATCH_SIZE' IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # THE TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, higher sizes might affect your scores.
    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS
    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 24  # DO NOT CHANGE THIS
    # Number of future time steps which are to be predicted.
    N_FUTURE = 24  # DO NOT CHANGE THIS
    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS
    # Code to create windowed train and validation datasets.
    # Complete the code in windowed_dataset.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    # Code to define your model.
    model = tf.keras.models.Sequential([
        Conv1D(filters=32,
               kernel_size=3,
               padding="causal",
               activation="relu",
               input_shape=[N_PAST, 7],
               ),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(16, return_sequences=True)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        tf.keras.layers.Dense(N_FEATURES)
    ])

    checkpoint_path = 'model/my_checkpoint.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1,
                                 )
    # Code to train and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        loss='mae',
        optimizer=optimizer,
        metrics=["mae"]
    )
    model.fit(
        train_set,
        validation_data=(valid_set),
        epochs=20,
        callbacks=[checkpoint],
    )
    model.load_weights(checkpoint_path)
    model.evaluate(valid_set)
    return model


if __name__ == '__main__':
    model = solution_model()
    # 0.053
    model.save("TF5_house_elec_power.h5")
