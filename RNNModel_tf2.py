import numpy as np
import pandas as pd

import os
import tensorflow as tf
from sklearn.metrics import f1_score

class RNNModel():
    # delete keep_prob, timesteps, num_layers, display_step, _lambd(we could consider to add ekrnel_regularization parameters)
    # num_classes == output_size
    def __init__(self, learning_rate=0.0001, n_epoch=5000, n_batch=700,
                 num_hidden=10, log_files_path=os.path.join(os.getcwd(), 'logs/'),
                 method='LSTM', output_size=3):

        self._learning_rate = learning_rate
        self._n_epoch = n_epoch
        self._n_batch = n_batch
        self._num_hidden = num_hidden
        self._log_files_path = log_files_path
        self._output_size = output_size

        self._method = method

        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None

        self._pred_train = None
        self._pred_test = None



    def _rnn(self, num_input):
        num_hidden = self._num_hidden

        method = self._method
        output_size = self._output_size

        # Please refer to https://www.tensorflow.org/guide/keras/rnn
        if method == 'LSTM':
            lstm_layer = tf.keras.layers.RNN(
                tf.keras.layers.LSTMCell(num_hidden),
                input_shape=(None, num_input))
            rnn_model = tf.keras.models.Sequential([lstm_layer,
                                                    tf.keras.layers.Dense(output_size)])

        # 2-layer LSTM, each layer has num_hidden units. And you can wrap more layers together by doing list comprehension.
        # Problem 1: Here in the original code, no activation function is designated
        if method == 'LSTMs':
            rnn_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_hidden,  return_sequences=True), input_shape=(None, num_input)),
                                                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_hidden)),
                                                    tf.keras.layers.Dense(output_size)])


        # 1-layer GRU
        if method == 'GRU':
            gru_layer = tf.keras.layers.RNN(
                tf.keras.layers.GRUCell(num_hidden),
                input_shape=(None, num_input))
            rnn_model = tf.keras.models.Sequential([gru_layer,
                                                    tf.keras.layers.Dense(output_size)])

        # 2-layer GRU
        if method == 'GRUs':
            rnn_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_hidden,  return_sequences=True), input_shape=(None, num_input)),
                                                    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_hidden)),
                                                    tf.keras.layers.Dense(output_size)])
        return rnn_model

    def train(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        n_batch = self._n_batch
        learning_rate = self._learning_rate
        n_epoch = self._n_epoch
        log_files_path = self._log_files_path
        num_input = x_train.shape[-1]

        model = self._rnn(num_input)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=["accuracy"])
        model.fit(x_train, y_train,
                  validation_data = (x_valid, y_valid),
                  batch_size = n_batch,
                  epochs = n_epoch)

        model.save(log_files_path)

        self._pred_train = model.predict(x_train)
        self._pred_test = model.predict(x_test)
        pred_test = self._pred_test

        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test

        return pred_test

    def evaluate(self, result_path=os.path.join(os.getcwd(), 'data')):

        pred_df_train = pd.DataFrame(self._pred_train, columns=['-1', '1'])
        pred_df_train['predict'] = pred_df_train.idxmax(axis=1)
        pred_df_train['true'] = pd.DataFrame(self._y_train, columns=['-1', '1']).idxmax(axis=1)

        # filename = os.path.join(result_path, 'pred_train_intc_up&down_nolesssprd.csv')
        # pred_df_train.to_csv(filename, index=False)

        pred_df_test = pd.DataFrame(self._pred_test, columns=['-1', '1'])
        pred_df_test['predict'] = pred_df_test.idxmax(axis=1)
        pred_df_test['true'] = pd.DataFrame(self._y_test, columns=['-1', '1']).idxmax(axis=1)

        filename = os.path.join(result_path, 'pred_test_intc_up&down_nolesssprd.csv')
        pred_df_test.to_csv(filename, index=False)

        train_f1 = f1_score(pred_df_train.true, pred_df_train.predict, average=None)
        test_f1 = f1_score(pred_df_test.true, pred_df_test.predict, average=None)
        print(train_f1)
        print(test_f1)

        return pred_df_train, pred_df_test

