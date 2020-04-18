import os
import time

import pandas as pd
import numpy as np
import tensorflow as tf
import features
import auto_features
import OButil as ob
from sklearn.model_selection import train_test_split
import SimpleStrategy2 as ss2


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


if __name__ == "__main__":
    # an example for the full flow
    code_size = 8
    encoder_layer_sizes = [32, 16]
    lag = 50
    quote_dir = './data/quote_intc_110816.csv'
    trade_dir = './data/trade_intc_110816.csv'
    out_order_book_filename = './data/order_book.csv'
    out_transaction_filename = './data/transaction.csv'

    #ob.preprocess_data(quote_dir, trade_dir, out_order_book_filename, out_transaction_filename)
    order_book_df = pd.read_csv(out_order_book_filename)
    transaction_df = pd.read_csv(out_transaction_filename)
    f = features.all_features(order_book_df, transaction_df, lag)[lag - 1:].ffill().bfill().reset_index(drop=True)
    auto_f = auto_features.auto_features(f.to_numpy(), code_size, encoder_layer_sizes,
                                         num_epochs=1, batch_size=4, display_step=1000)

    o = order_book_df[lag - 1:].to_numpy()
    t = transaction_df[lag - 1:].to_numpy()
    X = np.concatenate((t, auto_f, o), axis=1)
    X = pd.DataFrame(X)

    X, Y = ob.convert_to_dataset(X, window_size=10)
    X, Y = ob.over_sample(X, Y)

    start_time = time.time()

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)

    rnn = RNNModel(learning_rate=0.0001, n_epoch=5, batch_size=50,
                   num_hidden=32, method='LSTMs', output_size=3)

    rnn.train(train_X, train_Y, test_X, test_Y)

    # out of sample accuracy
    print("Out of sample accuracy:", (rnn.predict(test_X).argmax(1) == test_Y).mean())
    print("Total time: {0:.3f} seconds".format(time.time() - start_time))

    # test strategy, using the trained model rnn
    d = ss2.strategy_performance(rnn, order_book_df, transaction_df, window_size=10, mid_price_window=5, lag=50)
    ss2.plot(d)



