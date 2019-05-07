from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import time
import os
print(tf.__version__)

train_data=np.array([[0.0,0.0,0.0]])
model = keras.Sequential([
    keras.layers.Dense(3, activation='relu',input_shape=train_data[0].shape),
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_data, epochs=5)
