import os, sys, time, joblib
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import OButil, RNNModel_tf2, features, cnn_lstm


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


symbol = sys.argv[1]
model_type = sys.argv[2]
input_type = sys.argv[3]
mid_price_window = int(sys.argv[4])

learning_rate = 0.001
num_hidden = 64
use_pca = 'pca'

n_epoch = 150
batch_size = 128
lag = 50

output_dir = './logs/{}'.format(symbol)
file_prefix = '{}/{}_{}_{}'.format(output_dir, model_type, input_type, mid_price_window)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(file_prefix, exist_ok=True)

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
X = X.astype('float32')
if input_type in ['obfn', 'obn']:
    X[:, :, -20:] = OButil.OBnormal(X[:, :, -20:])
X, Y = OButil.over_sample(X, Y)

start_time = time.time()

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1)

if use_pca == 'pca':
    pca = PCA(n_components=0.95)
    ss = StandardScaler()
    train_X = transform_pc(train_X, pca, ss, train=True)
    valid_X = transform_pc(valid_X, pca, ss)
    test_X = transform_pc(test_X, pca, ss)
    joblib.dump(pca, '{}/pca.joblib'.format(file_prefix))
    joblib.dump(ss, '{}/ss.joblib'.format(file_prefix))


# Able to train CNNLSTM
if model_type == 'CNNLSTM':
    model = cnn_lstm.FullModel(learning_rate=learning_rate, num_hidden=num_hidden, leaky_relu_alpha=0.1, output_size=3)
    model.train(train_data=(train_X, train_Y), valid_data=(valid_X, valid_Y), num_epoch=n_epoch, batch_size=batch_size)
    print("Evaluating the model")
    model.evaluate(test_X, test_Y)

# Traditional RNN models
else:
    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size=batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size=batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size=batch_size)

    # Build model and train
    model = RNNModel_tf2.RNNModel(input_shape=train_X[0].shape, learning_rate=learning_rate,
                                  num_hidden=num_hidden, method=model_type, output_size=3,
                                  log_files_path=file_prefix)
    model.train(train_data, valid_data, n_epoch=n_epoch)

    print("Evaluating the model")
    model.evaluate(test_data)

print("Total time: {0:.3f} seconds".format(time.time() - start_time))
