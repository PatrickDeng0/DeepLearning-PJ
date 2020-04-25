from sklearn.model_selection import train_test_split

import RNNModel_tf2
import OButil
import features
import sys
import pandas as pd
import time
import tensorflow as tf
from sklearn.decomposition import PCA

# input_type:
# ob (original ob), obf (original ob and features),
# obn (normalized original ob), obfn (normalized ob and features)
# model_type:
# 'LSTM', 'LSTMs', 'GRU', 'GRUs'

# sample usage: python testing.py LSTMs,GRUs ob,obf,obn,obfn

_, model_types, input_types = sys.argv

model_types, input_types = model_types.split(','), input_types.split(',')

for model_type in model_types:
    for input_type in input_types:
        sys.stdout = open('./logs/{}_{}.txt'.format(model_type, input_type), 'w')
        n_epoch = 100
        batch_size = 512
        lag = 50

        order_book = pd.read_csv('./data/orderbook.csv')[lag - 1:].reset_index(drop=True)
        transaction = pd.read_csv('./data/transaction.csv')[lag - 1:].reset_index(drop=True)

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
        rnn = RNNModel_tf2.RNNModel(input_shape=train_X[0].shape, learning_rate=0.001, num_hidden=32, method=model_type,
                                    output_size=3, log_files_path='./logs/{}_{}'.format(model_type, input_type))
        rnn.train(train_data, valid_data, n_epoch=n_epoch)

        print("Evaluating the model")
        rnn.evaluate(test_data)

        print("Total time: {0:.3f} seconds".format(time.time() - start_time))

        sys.stdout.close()
