# Specify a simple NN generator
# Generates estimate of calendar month from the encoded latent space

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
                    units=250,
                    activation=tf.nn.elu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                tf.keras.layers.Dense(
                    units=250,
                    activation=tf.nn.elu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                tf.keras.layers.Dense(
                    units=250,
                    activation=tf.nn.elu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid),
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
        d1 = tf.math.squared_difference(mnth, x[2])
        d2 = tf.math.squared_difference(mnth + 1, x[2])
        d3 = tf.math.squared_difference(mnth - 1, x[2])
        d = tf.math.minimum(tf.math.minimum(d1, d2), d3)
        return tf.math.sqrt(tf.reduce_mean(d))

    @tf.function
    def train_on_batch(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss_values = self.compute_loss(x)
        gradients = tape.gradient(loss_values, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
