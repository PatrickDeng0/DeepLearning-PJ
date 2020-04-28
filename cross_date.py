import os, sys, time, joblib
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import OButil, RNNModel_tf2, features, cnn_lstm, train


def load_model(path):
    pca = joblib.load(path+'/pca.joblib')
    ss = joblib.load(path+'/ss.joblib')
    model = tf.keras.models.load_model(path)
    return pca, ss, model


def main():
    lag = 50
    batch_size = 128

    # Model Restore
    output_dir = './logs/{}'.format(symbol)
    file_prefix = '{}/{}_{}_midwin{}_win{}_rate{}'.format(
        output_dir, model_type, input_type, mid_price_window, x_window, learning_rate)
    pca, ss, model = load_model(file_prefix)

    # Data Preprocessing of test dates
    ob_file = './data/test_data/{}_ob_{}.csv'.format(symbol, test_date)
    trx_file = './data/test_data/{}_trx_{}.csv'.format(symbol, test_date)
    order_book = pd.read_csv(ob_file)
    transaction = pd.read_csv(trx_file)

    if input_type in ['obf', 'obfn']:
        f = features.all_features(order_book, transaction, lag)[lag - 1:]
        f.ffill(inplace=True)
        f.bfill(inplace=True)
        f.reset_index(drop=True, inplace=True)
        order_book = order_book[lag - 1:].reset_index(drop=True)
        transaction = transaction[lag - 1:].reset_index(drop=True)
        X = pd.concat([transaction, f, order_book], axis=1)
        del f
        del order_book
        del transaction
    else:
        X = pd.concat([transaction, order_book], axis=1)
        del order_book
        del transaction

    X, Y = OButil.convert_to_dataset(X, window_size=x_window, mid_price_window=mid_price_window)
    if input_type in ['obfn', 'obn']:
        X[:, :, -20:] = OButil.OBnormal(X[:, :, -20:])

    X = train.transform_pc(X, pca, ss)
    test_X = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size=batch_size)
    model.evaluate(test_X)


if __name__ == '__main__':
    symbol = sys.argv[1]
    model_type = sys.argv[2]
    input_type = sys.argv[3]
    mid_price_window = int(sys.argv[4])
    x_window = int(sys.argv[5])
    learning_rate = float(sys.argv[6])
    test_date = sys.argv[7]
    main()
