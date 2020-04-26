import os
import sys
import time

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import OButil
import RNNModel_tf2
import features

symbols = ['AAPL', 'AMD', 'GE', 'C']
learning_rates = [0.0001, 0.0005, 0.001]
nums_hidden = [32, 64]
model_types = ['LSTMs', 'GRUs']
input_types = ['ob', 'obn', 'obf', 'obfn']
n_epoch = 100
batch_size = 512
lag = 50

configs = np.array(np.meshgrid(symbols, nums_hidden, learning_rates, model_types, input_types)).T.reshape(-1, 5)

for (symbol, num_hidden, learning_rate, model_type, input_type) in configs:
    num_hidden = int(num_hidden)
    learning_rate = float(learning_rate)
    output_dir = './logs/{}_{}_{}'.format(symbol, num_hidden, learning_rate)
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = open('{}/{}_{}.log'.format(output_dir, model_type, input_type), 'w')

    ob_file = './data/{}_order_book.csv'.format(symbol)
    trx_file = './data/{}_transaction.csv'.format(symbol)

    order_book = pd.read_csv(ob_file)[lag - 1:].reset_index(drop=True)
    transaction = pd.read_csv(trx_file)[lag - 1:].reset_index(drop=True)

    if input_type in ['obf', 'obfn']:
        f = features.all_features(order_book, transaction, lag)[lag - 1:].ffill().bfill().reset_index(drop=True)
        pca = PCA(n_components=0.99)
        f = pd.DataFrame(pca.fit_transform(f))
        o = order_book[lag - 1:].reset_index(drop=True)
        t = transaction[lag - 1:].reset_index(drop=True)
        X = pd.concat([t, f, o], axis=1)
    else:
        X = pd.concat([transaction, order_book], axis=1)

    X, Y = OButil.convert_to_dataset(X, window_size=10, mid_price_window=1)
    if input_type in ['obfn', 'obn']:
        X[:, :, -20:] = OButil.OBnormal(X[:, :, -20:])
    X, Y = OButil.over_sample(X, Y)

    start_time = time.time()

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1)

    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size=batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size=batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size=batch_size)

    # Build model and train
    rnn = RNNModel_tf2.RNNModel(input_shape=train_X[0].shape, learning_rate=learning_rate,
                                num_hidden=num_hidden, method=model_type, output_size=3,
                                log_files_path='{}/{}_{}'.format(output_dir, model_type, input_type))
    rnn.train(train_data, valid_data, n_epoch=n_epoch)

    print("Evaluating the model")
    rnn.evaluate(test_data)

    print("Total time: {0:.3f} seconds".format(time.time() - start_time))

    sys.stdout.close()
