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

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""
# np.full([3,3],[[1,1,1],[1,1,1],[1,1,1]])
weights = []
for w in model.get_weights():
#    weights.append(np.tile(0,w.shape))
    weights.append(np.random.rand(*w.shape))
    print(w.shape)
#print(np.full((3,3),[[1,1,1],[1,1,1],[1,1,1]]))
"""
weights=[np.random.rand(*w.shape) for w in model.get_weights()]
model.set_weights(weights)
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)
"""
for w in model.get_weights():
    print(w.shape)
"""