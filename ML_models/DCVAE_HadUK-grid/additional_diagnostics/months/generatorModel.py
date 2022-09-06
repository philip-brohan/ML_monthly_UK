# Specify a simple NN generator
# Generates pdf of calendar month (12 probs, one per month)
#   from the encoded latent space

import os
import sys
import tensorflow as tf
import numpy as np


class NNG(tf.keras.Model):
    def __init__(self):
        super(NNG, self).__init__()
        self.latent_dim = 20

        self.diagnose_month = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(
                    units=100,
                    activation=tf.nn.elu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.1),
                ),
                tf.keras.layers.Dense(
                    units=100,
                    activation=tf.nn.elu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.1),
                ),
                tf.keras.layers.Dense(units=12, activation=None),
                tf.keras.layers.Softmax(),
            ]
        )

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def generate(self, z):
        mn = self.diagnose_month(z)
        return mn

    def call(self, x):
        latent = self.reparameterize(x[0], x[1])
        mnth = self.generate(latent)
        return mnth

    def compute_loss(self, x):
        latent = self.reparameterize(x[0], x[1])
        mnth = self.generate(latent)
        target = x[2]
        cce_MNTH = tf.keras.metrics.categorical_crossentropy(mnth, target)
        target = tf.roll(x[2], shift=1, axis=1)
        cce_MNTH += tf.keras.metrics.categorical_crossentropy(mnth, target) * 0.25
        target = tf.roll(x[2], shift=-1, axis=1)
        cce_MNTH += tf.keras.metrics.categorical_crossentropy(mnth, target) * 0.25
        cce_MNTH -= 16.11 * 0.5
        return cce_MNTH

    @tf.function
    def train_on_batch(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss_values = self.compute_loss(x)
        gradients = tape.gradient(loss_values, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
