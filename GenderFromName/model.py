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

# read data
df = pd.read_csv('name_gender.csv')


def pre_process(name):
    after_process_name = ''.join(re.split(r'[^A-Za-z]', name))
    after_process_name = after_process_name.lower()
    return after_process_name


# data cleaning
df['name'] = df['name'].apply(pre_process)

# Data Preparation
X = df['name']
y = df['gender']
for index, gender in enumerate(y):
    if gender == 'F':
        y[index] = 1
    else:
        y[index] = 0

# split to 3 Set
text_train, text_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
text_train, text_cv, y_train, y_cv = train_test_split(
    text_train, y_train, test_size=0.25, random_state=1)

# Tokenizer the word to char-level
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

# convert data to float64 to run in tensorflow
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_cv = X_cv.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
y_cv = y_cv.astype('float64')

# create model with 1st hidden lay 512 units, 2rd hidden lay 64. with L2 regularizers
model = Sequential()
model.add(Dense(512, input_dim=maxlen, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# epochs=60
hist = model.fit(X_train, y_train,
                 epochs=60,
                 verbose=False,
                 validation_data=(X_test, y_test),
                 batch_size=50)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Valuating Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_cv, y_cv, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist.history)


save_path = './Model'
model.save(save_path)
