import tensorflow as tf
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1, l2

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
from plot_keras_history import plot_history

df = pd.read_csv('name_gender.csv')


def pre_process(name):
    after_process_name = ''.join(re.split(r'[^A-Za-z]', name))
    after_process_name = after_process_name.lower()
    return after_process_name


df['name'] = df['name'].apply(pre_process)
names = df['name']

X = df['name']
y = df['gender']
for index, gender in enumerate(y):
    if gender == 'F':
        y[index] = 1
    else:
        y[index] = 0


text_train, text_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
text_train, text_cv, y_train, y_cv = train_test_split(
    text_train, y_train, test_size=0.25, random_state=1)
tokenizer = Tokenizer(num_words=None, lower=True,
                      char_level=True, oov_token='UNK')
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_test = tokenizer.texts_to_sequences(text_test)
X_cv = tokenizer.texts_to_sequences(text_cv)


word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
MAX_LEN = 20
maxlen = MAX_LEN

# Padding the sentences
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_cv = pad_sequences(X_cv, padding='post', maxlen=maxlen)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_cv = X_cv.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
model = Sequential([
    Dense(32, input_dim=maxlen, activation='relu'),
    Dense(1, activation='sigmoid')
])

# model compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          epochs=10,
          verbose=False,
          validation_data=(X_test, y_test))

save_path = '/Model'
model.save(save_path)
loaded_model = tf.keras.models.load_model(save_path)
word_represent = tokenizer.texts_to_sequences(['Mary'])
x_pred = pad_sequences(word_represent, padding='post', maxlen=maxlen)
print(loaded_model.predict(x_pred))
