import os

import tensorflow as tf


class RNNModel:
    # delete keep_prob, timesteps, num_layers, display_step, _lambd
    # we could consider to add ekrnel_regularization parameters
    # num_classes == output_size
    def __init__(self, learning_rate=0.0001, n_epoch=5000, batch_size=700,
                 num_hidden=10, log_files_path=os.path.join(os.getcwd(), 'logs'),
                 method='LSTM', output_size=3):

        self._learning_rate = learning_rate
        self._n_epoch = n_epoch
        self._batch_size = batch_size
        self._num_hidden = num_hidden
        self._log_files_path = log_files_path
        self._output_size = output_size
        self._method = method
        self._model = None

    def _rnn(self, input_shape):
        num_hidden = self._num_hidden

        method = self._method
        output_size = self._output_size

        if method == 'LSTM':
            lstm_layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_hidden), input_shape=input_shape)
            rnn_model = tf.keras.models.Sequential([lstm_layer, tf.keras.layers.Dense(output_size)])

        # 2-layer LSTM, each layer has num_hidden units. And you can wrap more layers together by doing list comprehension.
        # Problem 1: Here in the original code, no activation function is designated
        elif method == 'LSTMs':
            rnn_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(num_hidden, return_sequences=True), input_shape=input_shape),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_hidden)),
                tf.keras.layers.Dense(output_size)])

        # 1-layer GRU
        elif method == 'GRU':
            gru_layer = tf.keras.layers.RNN(tf.keras.layers.GRUCell(num_hidden), input_shape=input_shape)
            rnn_model = tf.keras.models.Sequential([gru_layer, tf.keras.layers.Dense(output_size)])

        # 2-layer GRU
        elif method == 'GRUs':
            rnn_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(num_hidden, return_sequences=True), input_shape=input_shape),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_hidden)),
                tf.keras.layers.Dense(output_size)])

        else:
            raise Exception("Invalid RNN method")

        return rnn_model

    def train(self, x_train, y_train, x_valid, y_valid):
        batch_size = self._batch_size
        learning_rate = self._learning_rate
        n_epoch = self._n_epoch
        log_files_path = self._log_files_path
        input_shape = x_train.shape[1:]

        self._model = self._rnn(input_shape)
        self._model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            metrics=["accuracy"])
        result = self._model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size,
                                 epochs=n_epoch)
        self._model.save(log_files_path)

        return result

    def predict(self, x_test):
        return self._model.predict(x_test)
