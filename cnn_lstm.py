import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Bidirectional, LSTM

import OButil as ob


def init_weights(shape, name):
    weight_initializer = tf.random_normal_initializer(stddev=(2.0 / np.product(shape[:-1])) ** 0.5)
    return tf.Variable(weight_initializer(shape), trainable=True, name=name, dtype=tf.float32)


def init_bias(shape, name, value=0.0):
    bias_initializer = tf.constant_initializer(value=value)
    return tf.Variable(bias_initializer(shape), trainable=True, name=name, dtype=tf.float32)


class ConvModule:
    def __init__(self, filter_shapes, leaky_relu_alpha):
        self.filter_shapes = filter_shapes
        self.leaky_relu_alpha = leaky_relu_alpha
        self.weights = [init_weights(s, "conv_w") for s in self.filter_shapes]
        self.biases = [init_bias((s[-1]), 'conv_b') for s in self.filter_shapes]

    def fwd(self, inp):
        c = tf.identity(inp)
        for s, w, b in zip(self.filter_shapes, self.weights, self.biases):
            c = tf.nn.conv2d(c, w, strides=[1, s[0], s[1], 1], padding='SAME')
            c = tf.nn.bias_add(c, b)
            c = tf.nn.leaky_relu(c, self.leaky_relu_alpha)
        return c

    def get_variables(self):
        return self.weights + self.biases


class InceptionModule:
    def __init__(self, leaky_relu_alpha):
        self.leaky_relu_alpha = leaky_relu_alpha
        self.net1_weight1 = init_weights([1, 1, 16, 32], 'inc_w11')
        self.net1_weight2 = init_weights([3, 1, 32, 32], 'inc_w12')
        self.net2_weight1 = init_weights([1, 1, 16, 32], 'inc_w21')
        self.net2_weight2 = init_weights([5, 1, 32, 32], 'inc_w22')
        self.net3_weight = init_weights([1, 1, 16, 32], 'inc_w3')
        self.net1_bias1 = init_bias((32), 'inc_b11')
        self.net1_bias2 = init_bias((32), 'inc_b12')
        self.net2_bias1 = init_bias((32), 'inc_b21')
        self.net2_bias2 = init_bias((32), 'inc_b22')
        self.net3_bias = init_bias((32), 'inc_b3')

    def _conv_layer(self, inp, weights, biases):
        ret = tf.nn.conv2d(inp, weights, strides=1, padding='SAME')
        ret = tf.nn.bias_add(ret, biases)
        return tf.nn.leaky_relu(ret, self.leaky_relu_alpha)

    def fwd(self, inp):
        n1 = self._conv_layer(inp, self.net1_weight1, self.net1_bias1)
        n1 = self._conv_layer(n1, self.net1_weight2, self.net1_bias2)
        n1 = tf.reshape(n1, [*n1.shape[:2], -1])

        n2 = self._conv_layer(inp, self.net2_weight1, self.net2_bias1)
        n2 = self._conv_layer(n2, self.net2_weight2, self.net2_bias2)
        n2 = tf.reshape(n2, [*n2.shape[:2], -1])

        n3 = tf.nn.max_pool(inp, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        n3 = self._conv_layer(n3, self.net3_weight, self.net3_bias)
        n3 = tf.reshape(n3, [*n3.shape[:2], -1])

        return tf.concat([n1, n2, n3], axis=2)

    def get_variables(self):
        return [
            self.net1_weight1,
            self.net1_weight2,
            self.net2_weight1,
            self.net2_weight2,
            self.net3_weight,
            self.net1_bias1,
            self.net1_bias2,
            self.net2_bias1,
            self.net2_bias2,
            self.net3_bias
        ]


class RNNModule:
    def __init__(self, num_hidden, output_size):
        self.model = None
        self.num_hidden = num_hidden
        self.output_size = output_size

    def fwd(self, inp):
        input_shape = inp.shape[1:]
        if self.model is None:
            self.model = tf.keras.models.Sequential([
                Bidirectional(LSTM(self.num_hidden, return_sequences=True), input_shape=input_shape),
                Bidirectional(LSTM(self.num_hidden)),
                Dense(self.output_size)
            ])

        return self.model(inp)

    def get_variables(self):
        return [] if self.model is None else self.model.trainable_variables


class FullModel:
    def __init__(self, leaky_relu_alpha, num_hidden, output_size):
        self.conv = ConvModule([[1, 2, 1, 16], [1, 2, 16, 16], [1, 5, 16, 16]], leaky_relu_alpha)
        self.inc = InceptionModule(leaky_relu_alpha)
        self.rnn = RNNModule(num_hidden, output_size)

    def fwd(self, order_book_batch, other_features):
        o = order_book_batch.reshape([*order_book_batch.shape, 1])
        c = self.conv.fwd(o)
        c = self.inc.fwd(c)
        merged = tf.concat([c, other_features], axis=2)
        return self.rnn.fwd(merged)

    def get_vars(self):
        return self.conv.get_variables() + self.inc.get_variables() + self.rnn.get_variables()


if __name__ == "__main__":
    out_order_book_filename = './data/order_book.csv'
    out_transaction_filename = './data/transaction.csv'
    # stored results from auto_features.py
    auto_features_filename = "./data/raw_features.csv"
    lag = 50

    order_book_df = pd.read_csv(out_order_book_filename)[lag - 1:].reset_index(drop=True)
    transaction_df = pd.read_csv(out_transaction_filename)[lag - 1:].reset_index(drop=True)
    f = pd.read_csv(auto_features_filename, index_col=0)[lag - 1:].reset_index(drop=True)

    X = pd.concat([transaction_df, f, order_book_df], axis=1)
    X, Y = ob.convert_to_dataset(X, window_size=10, mid_price_window=1)

    # Add orderbook normalize to X, replace the -20 to -1 location of X
    X[:,:,-20:] = ob.OBnormal(X[:,:,-20:])

    X, Y = ob.over_sample(X, Y)
    X = tf.cast(X, dtype=tf.float32).numpy()
    Y = tf.one_hot(Y, depth=3).numpy()
    leaky_alpha = 0.01
    learning_rate = 0.0001
    training_epochs = 500
    batch_size = 512

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
    splits = np.arange(batch_size, len(train_X), batch_size)
    train_X_batches = np.split(train_X, splits)
    train_Y_batches = np.split(train_Y, splits)

    m = FullModel(leaky_alpha, 64, 3)

    optimizer = tf.optimizers.Adam(learning_rate)

    results = np.zeros(training_epochs)
    start_time = time.time()
    for epoch in range(training_epochs):
        print("Starting epoch {}".format(epoch))
        for train_x, labels in zip(train_X_batches, train_Y_batches):
            with tf.GradientTape() as tape:
                loss = tf.nn.softmax_cross_entropy_with_logits(labels, m.fwd(train_x[:, :, -20:], train_x[:, :, :-20]))
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, m.get_vars())
            optimizer.apply_gradients(zip(grads, m.get_vars()))

        pred = np.argmax(m.fwd(test_X[:, :, :20], test_X[:, :, 20:]), axis=1)
        actual = np.argmax(test_Y, axis=1)
        acc = (pred == actual).mean()
        print("Out-of-sample accuracy: {}".format(acc))
        results[epoch] = acc

    print("Total time lapse: {0:.3f} seconds".format(time.time() - start_time))
    pd.DataFrame(results, columns=['acc']).to_csv('multimodal_results.csv')
