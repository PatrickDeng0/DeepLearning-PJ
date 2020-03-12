import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import OButil as ob
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def Build_Model(input_shape, num_nodes, output_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=num_nodes, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(units=num_nodes, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=output_shape, activation=tf.nn.softmax))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])
    return model


def main():
    data_dir = '../Data/'
    X, Y = ob.convert_to_dataset(data_dir+'orderbook.csv', window_size=10)

    # A lot of Y is 0: so actually we need 3 labels: 0 as down, 1 as remain, 2 as up
    Y = Y / np.abs(Y)
    Y = np.nan_to_num(Y, 0).astype(int) + 1

    # Reshape X for Oversampling, and then reshape back
    X = np.nan_to_num(X, 0)
    X_shape = X.shape
    X = X.reshape((X_shape[0], -1))
    model_RandomUnderSampler = RandomOverSampler(sampling_strategy='all')
    X, Y = model_RandomUnderSampler.fit_sample(X, Y)
    X = X.reshape((-1, X_shape[1], X_shape[2]))

    # Make batchs
    BATCH_SIZE = 512
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y))
    train_data = train_data.batch(batch_size=BATCH_SIZE)
    test_data = test_data.batch(batch_size=BATCH_SIZE)

    # Build model and train
    model = Build_Model(input_shape=train_X.shape[-2:], num_nodes=32, output_shape=3)
    model.summary()
    model.fit(train_data, validation_data=test_data, epochs=20)


if __name__ == '__main__':
    main()
