import os
import sys
import time

import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import OButil
import RNNModel_tf2
import features

symbol = sys.argv[1]
learning_rate = 0.001
num_hidden = 64
model_type = 'LSTMs'
input_type = 'ob'
use_pca = 'pca'
mid_price_windows = [1, 3, 5]

n_epoch = 150
batch_size = 128
lag = 50

for mid_price_window in mid_price_windows:
    output_dir = './logs/{}'.format(symbol)
    file_prefix = '{}/{}'.format(output_dir, mid_price_window)
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = open('{}.log'.format(file_prefix), 'w')

    ob_file = './data/{}_order_book.csv'.format(symbol)
    trx_file = './data/{}_transaction.csv'.format(symbol)

    order_book = pd.read_csv(ob_file)[lag - 1:].reset_index(drop=True)
    transaction = pd.read_csv(trx_file)[lag - 1:].reset_index(drop=True)

    if input_type in ['obf', 'obfn']:
        f = features.all_features(order_book, transaction, lag)[lag - 1:].ffill().bfill().reset_index(drop=True)
        o = order_book[lag - 1:].reset_index(drop=True)
        t = transaction[lag - 1:].reset_index(drop=True)
        X = pd.concat([t, f, o], axis=1)
    else:
        X = pd.concat([transaction, order_book], axis=1)

    X, Y = OButil.convert_to_dataset(X, window_size=10, mid_price_window=mid_price_window)
    if input_type in ['obfn', 'obn']:
        X[:, :, -20:] = OButil.OBnormal(X[:, :, -20:])
    X, Y = OButil.over_sample(X, Y)

    start_time = time.time()

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1)


    def transform_pc(train_x, pca_model, scaler, train=False):
        x_shape = train_x.shape
        if train:
            train_x = scaler.fit_transform(train_x.reshape(-1, x_shape[2]))
            train_x = pca_model.fit_transform(train_x)
        else:
            train_x = scaler.transform(train_x.reshape(-1, x_shape[2]))
            train_x = pca_model.transform(train_x)
        train_x = train_x.reshape(x_shape[0], x_shape[1], -1)
        return train_x


    if use_pca == 'pca':
        pca = PCA(n_components=0.99)
        ss = StandardScaler()
        train_X = transform_pc(train_X, pca, ss, train=True)
        valid_X = transform_pc(valid_X, pca, ss)
        test_X = transform_pc(test_X, pca, ss)

    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size=batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size=batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size=batch_size)

    # Build model and train
    rnn = RNNModel_tf2.RNNModel(input_shape=train_X[0].shape, learning_rate=learning_rate,
                                num_hidden=num_hidden, method=model_type, output_size=3,
                                log_files_path=file_prefix)
    rnn.train(train_data, valid_data, n_epoch=n_epoch)

    print("Evaluating the model")
    rnn.evaluate(test_data)

    print("Total time: {0:.3f} seconds".format(time.time() - start_time))

    sys.stdout.close()
