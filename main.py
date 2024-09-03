import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#handwritten digit dataset
mnist = tf.keras.datasets.mnist
#splits and load into train and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scaling down and normalizing data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#creating feed for nn
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=3)

accuracy, loss = model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('digits.keras')