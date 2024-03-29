import numpy as np

# (1) import를 해주세요
from tensorflow import keras
from keras.layers import Dense, LSTM, Lambda, Conv1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)

    # (2) 모델 정의 (Sequential)
    model = Sequential([
        Dense(3, input_shape=[1]),
        Dense(4),
        Dense(4),
        Dense(1),
    ])
    # (3) 컴파일 (compile)
    model.compile(optimizer='sgd', loss="mse")
    # (4) 학습 (fit)
    model.fit(xs, ys, epochs=1000)
    model.evaluate(xs, ys)
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
