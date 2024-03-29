# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#

# =================================================== #

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 2 - mnist
# val_loss: 0.07
# val_acc: 0.97
# =================================================== #
# =================================================== #


import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, Dropout, Reshape, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


def solution_model():
    mnist = tf.keras.datasets.mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    print()

    x_train = x_train / (x_train.max())
    x_valid = x_valid / (x_valid.max())

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    check_path = "asdf.ckpg"
    check_point = ModelCheckpoint(
        filepath=check_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1)

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=20, callbacks=[check_point])
    model.load_weights(check_path)
    model.evaluate(x_valid, y_valid)
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-mnist.h5")