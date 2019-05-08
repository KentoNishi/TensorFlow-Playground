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
from random import randint
print(tf.__version__)

# testInputs= np.array([np.array(np.random.uniform(0,1,10), dtype='float64') for k in range(100)])
testInputs=np.array([np.array([0,1,2,3], dtype='float64'),np.array([3,2,1,0], dtype='float64')])
testLabels=testInputs
model = keras.Sequential([
    keras.layers.Dense(4,activation='tanh'),
    keras.layers.Dense(4,activation='tanh')
])
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
weights=[np.random.rand(*w.shape) for w in model.get_weights()]
model.set_weights(weights)
model.fit(testInputs, testLabels, epochs=100000)
test_loss, test_acc = model.evaluate(testInputs, testLabels)
print('\nTest accuracy:', test_acc)