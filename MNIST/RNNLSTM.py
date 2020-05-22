import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, SimpleRNN, GRU, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train/255.0, X_test/255.0
i = Input(shape = X_train[0].shape)
x = LSTM(128)(i)
x = Dense(10, activation = 'softmax')(x)
model = Model(i, x)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
r = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10)

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label = 'Loss')
plt.plot(r.history['val_loss'], label = 'Val_Loss')
plt.legend()

plt.plot(r.history['accuracy'], label = 'Accuracy')
plt.plot(r.history['val_accuracy'], label = 'Val_Accuracy')
plt.legend()