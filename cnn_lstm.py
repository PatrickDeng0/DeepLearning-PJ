import time
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Bidirectional, LSTM


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
                Bidirectional(LSTM(self.num_hidden), input_shape=input_shape),
                Dense(self.num_hidden, activation='relu'),
                Dense(self.output_size, activation='softmax')
            ])

        return self.model(inp)

    def get_variables(self):
        return [] if self.model is None else self.model.trainable_variables


class FullModel:
    def __init__(self, learning_rate, num_hidden=3, leaky_relu_alpha=0.1, output_size=3):
        self.conv = ConvModule([[1, 2, 1, 16], [1, 2, 16, 16], [1, 5, 16, 16]], leaky_relu_alpha)
        self.inc = InceptionModule(leaky_relu_alpha)
        self.rnn = RNNModule(num_hidden, output_size)
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def fwd(self, X):
        order_book_batch = X[:, :, -20:]
        other_features = X[:, :, :-20]
        o = order_book_batch.reshape([*order_book_batch.shape, 1])
        c = self.conv.fwd(o)
        c = self.inc.fwd(c)
        merged = tf.concat([c, other_features], axis=2)
        return self.rnn.fwd(merged)

    def get_vars(self):
        return self.conv.get_variables() + self.inc.get_variables() + self.rnn.get_variables()

    def predict(self, X):
        return np.argmax(self.fwd(X), axis=1)

    def evaluate(self, X, Y):
        pred = self.predict(X)
        acc = (pred == Y).mean()
        return acc

    # train_data: tf.dataset object
    # valid_data: numpy object. For convenience in strategy
    def train(self, train_data, valid_data=None, num_epoch=100, batch_size=128, class_weights=None):
        train_X, train_Y = train_data
        shuffled_X, shuffled_Y = shuffle(train_X, train_Y)
        val_acc_list = []
        start_time = time.time()

        for epoch in range(num_epoch):
            for batch_X, batch_Y in get_batch(shuffled_X, shuffled_Y, batch_size):
                with tf.GradientTape() as tape:
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(batch_Y, self.fwd(batch_X))
                    if class_weights is not None:
                        weights = np.array(list(map(lambda x: class_weights[x], batch_Y)))
                        loss = tf.multiply(loss, weights)
                    loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, self.get_vars())
                self.optimizer.apply_gradients(zip(grads, self.get_vars()))

            if valid_data is not None:
                valid_X, valid_Y = valid_data
                val_acc = self.evaluate(valid_X, valid_Y)
                val_acc_list.append(val_acc)
                print("Epoch {}: Train acc: {}, Validation acc: {}".
                      format(epoch, self.evaluate(train_X, train_Y), val_acc))
                if len(val_acc_list) >= 6 and np.array(val_acc_list[-6:]).argmax() == 0:
                    print("Early stopping")
                    break

        print("Total time lapse: {0:.3f} seconds".format(time.time() - start_time))


def get_batch(X, Y, batch_size):
    nums = X.shape[0]
    for i in range(nums // batch_size):
        yield X[i * batch_size: (i + 1) * batch_size], Y[i * batch_size: (i + 1) * batch_size]
