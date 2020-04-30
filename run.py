import joblib
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cnn_lstm
import features
import ob_util
import rnn_model

num_hidden = 64
n_epoch = 150
batch_size = 256
lag = 50


def transform_pc(train_x, pca_model, scaler, training=False):
    x_shape = train_x.shape
    if training:
        train_x = scaler.fit_transform(train_x.reshape(-1, x_shape[2]))
        train_x = pca_model.fit_transform(train_x)
    else:
        train_x = scaler.transform(train_x.reshape(-1, x_shape[2]))
        train_x = pca_model.transform(train_x)
    train_x = train_x.reshape(x_shape[0], x_shape[1], -1)
    return train_x


def load_data(raw_data_directory, gen_features=True):
    num_files = len([fn for fn in os.listdir(raw_data_directory) if '.csv' in fn]) / 2

    data_list = []
    for i in range(int(num_files)):
        order_book = pd.read_csv('{}/ob_{}.csv'.format(raw_data_directory, i))
        if gen_features:
            transaction = pd.read_csv('{}/trx_{}.csv'.format(raw_data_directory, i))
            data_list.append(features.all_features(order_book, transaction, lag, include_ob=True))
        else:
            data_list.append(order_book)

    return data_list


def convert_data(data_list, x_window, mid_price_window):
    X, Y = None, None
    for data in data_list:
        x, y = ob_util.convert_to_dataset(data, window_size=x_window, mid_price_window=mid_price_window)
        if X is None or Y is None:
            X, Y = x, y
        else:
            X = np.concatenate([X, x])
            Y = np.concatenate([Y, y])
    return X, Y


def get_class_weight(Y):
    results, counts = np.unique(Y, return_counts=True)
    max_count = np.max(counts)
    ret = {}
    for i in range(len(results)):
        ret[results[i]] = max_count / counts[i]
    print("Class weights are:", ret)
    return ret


def train(X, Y, model_type, learning_rate, file_prefix):
    class_weight = get_class_weight(Y)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1)

    start_time = time.time()

    if model_type == 'CNNLSTM':
        # only take the original order book columns
        train_X = train_X[:, :, -20:]
        test_X = test_X[:, :, -20:]
        valid_X = valid_X[:, :, -20:]

        model = cnn_lstm.FullModel(learning_rate=learning_rate, num_hidden=num_hidden,
                                   leaky_relu_alpha=0.1, output_size=3)
        model.train(train_data=(train_X, train_Y), class_weights=class_weight,
                    valid_data=(valid_X, valid_Y), num_epoch=n_epoch, batch_size=batch_size)
        val_acc = model.evaluate(test_X, test_Y)
        print("Evaluating the model, acc:", val_acc)
    else:
        pca_model = PCA(n_components=0.95)
        ss_model = StandardScaler()
        train_X = transform_pc(train_X, pca_model, ss_model, training=True)
        valid_X = transform_pc(valid_X, pca_model, ss_model)
        test_X = transform_pc(test_X, pca_model, ss_model)
        joblib.dump(pca_model, '{}/pca.joblib'.format(file_prefix))
        joblib.dump(ss_model, '{}/ss.joblib'.format(file_prefix))

        train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size=batch_size)
        valid_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size=batch_size)
        test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size=batch_size)

        # Build model and train
        model = rnn_model.RNNModel(input_shape=train_X[0].shape, learning_rate=learning_rate,
                                   num_hidden=num_hidden, method=model_type, output_size=3,
                                   log_files_path=file_prefix)
        model.train(train_data, valid_data, n_epoch=n_epoch, class_weight=class_weight)

        val_acc = model.evaluate(test_data)[1]

    print("Total training time: {0:.3f} seconds".format(time.time() - start_time))
    return val_acc


def parse_config(config):
    return config[0], int(config[1]), int(config[2]), float(config[3])


def gen_file_prefix(o_dir, model_type, mid_price_window, x_window, learning_rate):
    return '{}/{}_midwin{}_xwin{}_rate{}'.format(o_dir, model_type, mid_price_window, x_window, learning_rate)


def load_model(path):
    pca_model = joblib.load(path + '/pca.joblib')
    ss_model = joblib.load(path + '/ss.joblib')
    trained_model = tf.keras.models.load_model(path)
    return pca_model, ss_model, trained_model


if __name__ == '__main__':
    # mode: single, grid, test
    mode = sys.argv[1]
    if mode not in ['train_grid', 'train_single', 'test']:
        sys.exit('Mode should be one of \'train_grid\', \'train_single\', \'test\'')

    symbol = sys.argv[2]
    raw_data_dir = './data/{}'.format(symbol)
    out_dir = './logs/{}'.format(symbol)
    os.makedirs(out_dir, exist_ok=True)

    model_types = sys.argv[3].split(',')
    mid_price_windows = sys.argv[4].split(',')
    x_windows = sys.argv[5].split(',')
    learning_rates = sys.argv[6].split(',')

    if mode == 'train_grid':
        data_set = load_data(raw_data_dir, (np.array(model_types) != 'CNNLSTM').any())
        configs = np.array(np.meshgrid(model_types, mid_price_windows, x_windows, learning_rates)).T.reshape(-1, 4)
        train_result = {}
        for cfg in configs:
            mod_type, mid_win, x_win, learn_rate = parse_config(cfg)
            f_prefix = gen_file_prefix(out_dir, mod_type, mid_win, x_win, learn_rate)
            os.makedirs(f_prefix, exist_ok=True)
            x_tensor, y_tensor = convert_data(data_set, x_win, mid_win)
            train_result[tuple(cfg)] = train(x_tensor, y_tensor, mod_type, learn_rate, f_prefix)
        best_acc, best_cfg = max([(val_acc, cfg) for cfg, val_acc in train_result.items()])
        print("The best training result is {0:.4f}, the config is {1}".format(best_acc, best_cfg))

    else:
        mod_type = model_types[0]
        mid_win = int(mid_price_windows[0])
        x_win = int(x_windows[0])
        learn_rate = float(learning_rates[0])
        f_prefix = gen_file_prefix(out_dir, mod_type, mid_win, x_win, learn_rate)

        if mode == 'train_single':
            data_set = load_data(raw_data_dir, mod_type != 'CNNLSTM')
            os.makedirs(f_prefix, exist_ok=True)
            x_tensor, y_tensor = convert_data(data_set, x_win, mid_win)
            train(x_tensor, y_tensor, mod_type, learn_rate, f_prefix)

        elif mode == 'test':
            test_date = sys.argv[7]
            pca, ss, mod = load_model(f_prefix)
            ob_file = './data/test_data/{}_ob_{}.csv'.format(symbol, test_date)
            trx_file = './data/test_data/{}_trx_{}.csv'.format(symbol, test_date)
            ob = pd.read_csv(ob_file)
            if mod_type == 'CNNLSTM':
                test_x, test_y = ob_util.convert_to_dataset(ob, window_size=x_win, mid_price_window=mid_win)
            else:
                trx = pd.read_csv(trx_file)
                test_x = features.all_features(ob, trx, lag, include_ob=True)
                test_x, test_y = ob_util.convert_to_dataset(test_x, window_size=x_win, mid_price_window=mid_win)
                test_x = transform_pc(test_x, pca, ss)
            test_input = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size=batch_size)
            mod.evaluate(test_input, verbose=2)
