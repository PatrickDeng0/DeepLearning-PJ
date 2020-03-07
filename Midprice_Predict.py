import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import OButil as ob
from sklearn.model_selection import train_test_split


def Build_Model(num_nodes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=num_nodes))
    model.add(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))
    model.compile(optimizer=tf.optimizers.Adam, loss=tf.losses.BinaryCrossentropy, metrics=['Accuracy'])
    return model


