import os, sys, time, joblib
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ob_util, rnn_model, features, cnn_lstm


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


def main():
    num_hidden = 64

    n_epoch = 150
    batch_size = 256
    lag = 50

    output_dir = './logs/{}'.format(symbol)
    file_prefix = '{}/{}_{}_midwin{}_xwin{}_rate{}'.format(
        output_dir, model_type, input_type, mid_price_window, x_window, learning_rate)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(file_prefix, exist_ok=True)

    raw_data_dir = './data/{}'.format(symbol)
    num_files = len([fn for fn in os.listdir(raw_data_dir) if '.csv' in fn]) / 2

    X, Y = None, None
    for i in range(int(num_files)):
        order_book = pd.read_csv('{}/ob_{}.csv'.format(raw_data_dir, i))
        transaction = pd.read_csv('{}/trx_{}.csv'.format(raw_data_dir, i))

        if input_type in ['obf', 'obfn']:
            f = features.all_features(order_book, transaction, lag)[lag - 1:]
            f.ffill(inplace=True)
            f.bfill(inplace=True)
            f.reset_index(drop=True, inplace=True)
            order_book = order_book[lag - 1:].reset_index(drop=True)
            transaction = transaction[lag - 1:].reset_index(drop=True)
            x = pd.concat([transaction, f, order_book], axis=1)
            del f
            del order_book
            del transaction
        else:
            x = pd.concat([transaction, order_book], axis=1)
            del order_book
            del transaction

        x, y = ob_util.convert_to_dataset(x, window_size=x_window, mid_price_window=mid_price_window)
        if X is None or Y is None:
            X, Y = x, y
        else:
            X = np.concatenate([X, x])
            Y = np.concatenate([Y, y])


    if input_type in ['obfn', 'obn']:
        X[:, :, -20:] = ob_util.OBnormal(X[:, :, -20:])

    # Instead of Oversample, we use different loss weight to balance the loss function
    # X, Y = ob_util.over_sample(X, Y)
    results, counts = np.unique(Y, return_counts=True)
    max_count = np.max(counts)
    class_weight = {}
    for i in range(len(results)):
        class_weight[results[i]] = max_count / counts[i]
    print(class_weight)

    start_time = time.time()

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1)

    # only apply pca when features are included
    if input_type in ['obf', 'obfn']:
        pca = PCA(n_components=0.95)
        ss = StandardScaler()
        train_X = transform_pc(train_X, pca, ss, train=True)
        valid_X = transform_pc(valid_X, pca, ss)
        test_X = transform_pc(test_X, pca, ss)
        joblib.dump(pca, '{}/pca.joblib'.format(file_prefix))
        joblib.dump(ss, '{}/ss.joblib'.format(file_prefix))

    if model_type == 'CNNLSTM':
        model = cnn_lstm.FullModel(learning_rate=learning_rate, num_hidden=num_hidden, leaky_relu_alpha=0.1,
                                   output_size=3)
        model.train(train_data=(train_X, train_Y), valid_data=(valid_X, valid_Y), num_epoch=n_epoch,
                    batch_size=batch_size)
        print("Evaluating the model, acc:", model.evaluate(test_X, test_Y))

    # Traditional RNN models
    else:
        train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size=batch_size)
        valid_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size=batch_size)
        test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size=batch_size)

        # Build model and train
        model = rnn_model.RNNModel(input_shape=train_X[0].shape, learning_rate=learning_rate,
                                   num_hidden=num_hidden, method=model_type, output_size=3,
                                   log_files_path=file_prefix)
        model.train(train_data, valid_data, n_epoch=n_epoch, class_weight=class_weight)

        print("Evaluating the model")
        model.evaluate(test_data)

    print("Total time: {0:.3f} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    symbol = sys.argv[1]
    model_type = sys.argv[2]
    input_type = sys.argv[3]
    mid_price_window = int(sys.argv[4])
    x_window = int(sys.argv[5])
    learning_rate = float(sys.argv[6])
    main()
