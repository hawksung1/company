import json
import tensorflow as tf
import numpy as np
import urllib

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

# data load
with open('sarcasm.json') as f:
    datas = json.load(f)

sentences = []
labels = []
for data in datas:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])

training_size = 20000
train_sentences = sentences[:training_size]
train_labels = labels[:training_size]
validation_sentences = sentences[training_size:]
validation_labels = labels[training_size:]

# preprocessing
vocab_size = 1000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
word_index = tokenizer.word_index

# 한 문장의 최대 단어 숫자
max_length = 120
# 잘라낼 문장의 위치
trunc_type='post'
# 채워줄 문장의 위치
padding_type='post'

# sequence length set
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

# modeling
embedding_dim = 20
x = Embedding(vocab_size, embedding_dim, input_length=max_length)
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# check point
checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

# train
epochs=10
history = model.fit(train_padded, train_labels,
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                    epochs=epochs)
model.load_weights(checkpoint_path)
# eval
model.evaluate(validation_padded, validation_labels)
model.save("TF4-sarcasm-00.h5")