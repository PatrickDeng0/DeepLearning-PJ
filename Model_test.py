
from RNNModel_tf2 import RNNModel
import numpy as np
import pandas as pd

import os
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    model = RNNModel(n_epoch=3, method="GRUs", output_size=10)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_x, test_x = train_test_split(x_test, test_size=.2, random_state=1)
    train_x, val_x = train_test_split(train_x, test_size=.2, random_state=1)

    train_y, test_y = train_test_split(y_test, test_size=.2, random_state=1)
    train_y, val_y = train_test_split(train_y, test_size=.2, random_state=1)
    print((len(train_x), len(val_x), len(test_x)))
    print((len(train_y), len(val_y), len(test_y)))

    model.train(train_x, train_y, val_x, val_y, test_x, test_y)





