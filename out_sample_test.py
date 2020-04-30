import joblib
import sys

import pandas as pd
import tensorflow as tf

import features
import ob_util
import train


def load_model(path):
    pca = joblib.load(path + '/pca.joblib')
    ss = joblib.load(path + '/ss.joblib')
    model = tf.keras.models.load_model(path)
    return pca, ss, model


def main():
    lag = 50
    batch_size = 128

    # Model Restore
    output_dir = './logs/{}'.format(symbol)
    file_prefix = '{}/{}_{}_midwin{}_win{}_rate{}_{}'.format(
        output_dir, model_type, input_type, mid_price_window, x_window, learning_rate, use_pca)
    pca, ss, model = load_model(file_prefix)

    # Data Preprocessing of test dates
    ob_file = './data/test_data/{}_ob_{}.csv'.format(symbol, test_date)
    trx_file = './data/test_data/{}_trx_{}.csv'.format(symbol, test_date)
    order_book = pd.read_csv(ob_file)
    transaction = pd.read_csv(trx_file)

    if input_type in ['obf', 'obfn']:
        X = features.all_features(order_book, transaction, lag, include_ob=True)
    else:
        X = pd.concat([transaction, order_book], axis=1)

    X, Y = ob_util.convert_to_dataset(X, window_size=x_window, mid_price_window=mid_price_window)
    if input_type in ['obfn', 'obn']:
        X[:, :, -20:] = ob_util.normalize_ob(X[:, :, -20:])

    if use_pca == 'pca':
        X = train.transform_pc(X, pca, ss)

    test_X = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size=batch_size)
    model.evaluate(test_X, verbose=2)


if __name__ == '__main__':
    symbol = sys.argv[1]
    model_type = sys.argv[2]
    input_type = sys.argv[3]
    mid_price_window = int(sys.argv[4])
    x_window = int(sys.argv[5])
    learning_rate = float(sys.argv[6])
    use_pca = sys.argv[7]
    test_date = sys.argv[8]
    main()
