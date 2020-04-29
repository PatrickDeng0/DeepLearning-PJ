import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


class RNNModel:
    def __init__(self, input_shape, learning_rate=0.001, num_hidden=64,
                 log_files_path=os.path.join(os.getcwd(), 'logs'),
                 method='LSTM', output_size=3):

        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._num_hidden = num_hidden
        self._log_files_path = log_files_path
        self._output_size = output_size
        self._method = method
        self._model = self._rnn()

    def _rnn(self):
        model = tf.keras.Sequential()
        # 1-layer LSTM
        if self._method == 'LSTM':
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self._num_hidden),
                                                    input_shape=self._input_shape))
            model.add(tf.keras.layers.Dense(units=self._num_hidden, activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(units=self._output_size, activation=tf.nn.softmax))

        # 1-layer LSTMs
        elif self._method == 'LSTMs':
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self._num_hidden, return_sequences=True),
                                                    input_shape=self._input_shape))
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self._num_hidden)))
            model.add(tf.keras.layers.Dense(units=self._num_hidden, activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(self._output_size, activation=tf.nn.softmax))

        # 1-layer GRU
        elif self._method == 'GRU':
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=self._num_hidden),
                                                    input_shape=self._input_shape))
            model.add(tf.keras.layers.Dense(units=self._num_hidden, activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(units=self._output_size, activation=tf.nn.softmax))

        # 2-layer GRUs
        elif self._method == 'GRUs':
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self._num_hidden, return_sequences=True),
                                                    input_shape=self._input_shape))
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self._num_hidden)))
            model.add(tf.keras.layers.Dense(units=self._num_hidden, activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(self._output_size, activation=tf.nn.softmax))
        else:
            raise Exception("Invalid RNN method")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['sparse_categorical_accuracy'])
        return model

    def train(self, train_data, valid_data, n_epoch, class_weight):
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=2)
        log_files_path = self._log_files_path
        history = self._model.fit(train_data, validation_data=valid_data, epochs=n_epoch, verbose=2,
                                  callbacks=[es], class_weight=class_weight)
        self._model.save(log_files_path)
        return history

    def evaluate(self, test_data):
        return self._model.evaluate(test_data, verbose=2)

    def predict(self, test_X):
        return self._model.predict(test_X)
