from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import feature_column
from tensorflow.keras import layers
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import os
import random
import subprocess
print(tf.__version__)

(inputs, labels) = ([1,2,3],[0,1,2])
(testInputs, testLabels) = (inputs,labels)
model = keras.Sequential([
    keras.layers.Dense(1,activation='relu',input_shape=(1,)),
    keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
weights=[np.random.rand(*w.shape) for w in model.get_weights()]
model.set_weights(weights)
model.fit(inputs, labels, epochs=1000)
test_loss, test_acc = model.evaluate(testInputs, testLabels)
print('\nTest accuracy:', test_acc)