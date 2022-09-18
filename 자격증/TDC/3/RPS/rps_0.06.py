# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# For this task you will build a classifier for Rock-Paper-Scissors
# based on the rps dataset.
#
# IMPORTANT: Your final layer should be as shown, do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail.
#
# NOTE THAT THIS IS UNLABELLED DATA.
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - rps
# val_loss: 0.0871
# val_acc: 0.97
# =================================================== #
# =================================================== #

import urllib.request
import zipfile
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Dropout, Reshape, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from keras.applications import efficientnet_v2


url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip')
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

TRAINING_DIR = "tmp/rps/"
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=32,
                                                        target_size=(150, 150),
                                                        class_mode='categorical',
                                                        subset='training',
                                                        )
validation_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                            batch_size=32,
                                                            target_size=(150, 150),
                                                            class_mode='categorical',
                                                            subset='validation',
                                                            )
transfer_model = efficientnet_v2.EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
transfer_model.trainable = False

model = tf.keras.models.Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
checkpoint_path = "tmp_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              save_best_only=True,
                              monitor='val_loss',
                              verbose=1)
model.fit(train_generator,
          validation_data=(validation_generator),
          epochs=25,
          callbacks=[checkpoint],
          )
model.load_weights(checkpoint_path)
model.evaluate(validation_generator)

model.save("TF3-rps-0.06.h5")