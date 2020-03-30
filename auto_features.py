import tensorflow as tf
import numpy as np


class FullyConnectedAutoEncoder(tf.keras.Model):
    def __init__(self, original_dim, code_size, encoder_layer_sizes):
        super(FullyConnectedAutoEncoder, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.encoder = []
        self.decoder = []

        for s in encoder_layer_sizes:
            self.encoder.append(tf.keras.layers.Dense(s, activation=tf.nn.relu))

        self.bottleneck = tf.keras.layers.Dense(code_size, activation=tf.nn.relu)

        for s in reversed(encoder_layer_sizes):
            self.decoder.append(tf.keras.layers.Dense(s, activation=tf.nn.relu))

        self.output_layer = tf.keras.layers.Dense(original_dim)

    def call(self, input_data):
        x = self.flatten_layer(input_data)
        x_norm, x_bar = self.batch_norm_layer(x, training=True), self.batch_norm_layer(x, training=True)

        for encoder in self.encoder:
            x_bar = encoder(x_bar)
        x_bar = self.bottleneck(x_bar)
        for decoder in self.decoder:
            x_bar = decoder(x_bar)

        x_bar = self.output_layer(x_bar)

        return x_bar, x_norm

    def encode(self, input_data):
        x = self.flatten_layer(input_data)
        x = self.batch_norm_layer(x, training=False)
        for encoder in self.encoder:
            x = encoder(x)

        return self.bottleneck(x)


def loss(y, y_bar):
    return tf.losses.mean_squared_error(y, y_bar)


def grad(model, inputs):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped = model(inputs)
        loss_value = loss(inputs_reshaped, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction


def auto_features(x, code_size, encoder_layer_sizes, learning_rate=0.001, num_epochs=50, batch_size=4, display_step=20):
    m = FullyConnectedAutoEncoder(np.prod(x.shape[1:]), code_size, encoder_layer_sizes)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    global_step = 0
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        for i in range(0, len(x), batch_size):
            x_input = x[i: i + batch_size]
            loss_value, grads, inputs_reshaped, reconstruction = grad(m, x_input)
            optimizer.apply_gradients(zip(grads, m.trainable_variables), str(global_step))

            if global_step % display_step == 0:
                print("Step: {}, Loss: {}".format(global_step, loss(inputs_reshaped, reconstruction).numpy()))

            global_step += 1

    return m.encode(x)
