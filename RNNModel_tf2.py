import os
import time

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import OButil as ob
import SimpleStrategy2 as ss2


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
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=2)
        log_files_path = self._log_files_path
        history = self._model.fit(train_data, validation_data=valid_data, epochs=n_epoch, verbose=2,
                                  callbacks=[es], class_weight=class_weight)
        self._model.save(log_files_path)
        return history

    def evaluate(self, test_data):
        return self._model.evaluate(test_data)

    def predict(self, test_X):
        return self._model.predict(test_X)


def main():
    n_epoch = 50
    batch_size = 128
    lag = 50

    quote_dir = './data/quote_intc_110816.csv'
    trade_dir = './data/trade_intc_110816.csv'
    out_order_book_filename = './data/order_book.csv'
    out_transaction_filename = './data/transaction.csv'
    auto_features_filename = "./data/raw_features.csv"

    order_book_df = pd.read_csv(out_order_book_filename)[lag - 1:].reset_index(drop=True)
    transaction_df = pd.read_csv(out_transaction_filename)[lag - 1:].reset_index(drop=True)
    f = pd.read_csv(auto_features_filename, index_col=0)[lag - 1:].reset_index(drop=True)
    X = pd.concat([transaction_df, f, order_book_df], axis=1)

    # Convert to dataset
    X, Y = ob.convert_to_dataset(X, window_size=10, mid_price_window=3)
    # X[:, :, -20:] = ob.OBnormal(X[:, :, -20:])
    X, Y = ob.over_sample(X, Y)

    # Counting time
    start_time = time.time()

    # Split train, valid, test and Make batchs
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1)

    # Preprocessing
    pca = PCA(n_components=0.95)
    ss = StandardScaler()

    def Preprocessing(train_X, pca, ss, train=False):
        X_shape = train_X.shape
        if train:
            train_X = ss.fit_transform(train_X.reshape(-1, X_shape[2]))
            train_X = pca.fit_transform(train_X)
        else:
            train_X = ss.transform(train_X.reshape(-1, X_shape[2]))
            train_X = pca.transform(train_X)
        train_X = train_X.reshape(X_shape[0], X_shape[1], -1)
        return train_X

    train_X = Preprocessing(train_X, pca, ss, train=True)
    valid_X = Preprocessing(valid_X, pca, ss)
    test_X = Preprocessing(test_X, pca, ss)

    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size=batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size=batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size=batch_size)

    # Build model and train
    rnn = RNNModel(input_shape=train_X[0].shape, learning_rate=0.001, num_hidden=64,
                   method='LSTM', output_size=3)
    rnn.train(train_data, valid_data, n_epoch=n_epoch)

    rnn.evaluate(test_data)

    print("Total time: {0:.3f} seconds".format(time.time() - start_time))

    # testing strategy performance
    d = ss2.strategy_performance(rnn, order_book_df, transaction_df, window_size=10, mid_price_window=1, lag=50)
    ss2.plot(d)

    return rnn


if __name__ == '__main__':
    main()
