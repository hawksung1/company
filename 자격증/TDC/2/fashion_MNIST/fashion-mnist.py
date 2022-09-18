# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. YOur input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 2 - fashion mnist
# val_loss: 0.33
# val_acc: 0.89
# =================================================== #
# =================================================== #


import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, Dropout, Reshape, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()
# YOUR CODE HERE
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

model.save("TF2-fashion-mnist.h5")