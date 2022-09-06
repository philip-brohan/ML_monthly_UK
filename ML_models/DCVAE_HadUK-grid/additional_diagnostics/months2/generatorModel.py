# Specify a simple NN generator
# Generates pdf of calendar month (12 probs, one per month)
#   from the 4 weather fields

import os
import sys
import tensorflow as tf
import numpy as np


class NNG(tf.keras.Model):
    def __init__(self):
        super(NNG, self).__init__()

        self.diagnose_month = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(1440, 896, 4)),
                tf.keras.layers.Conv2D(
                    filters=5 * 4,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2D(
                    filters=5 * 2,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2D(
                    filters=10,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2D(
                    filters=40,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    units=24,
                    activation=tf.nn.elu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.1),
                ),
                tf.keras.layers.Dense(units=12, activation=None),
                tf.keras.layers.Softmax(),
            ]
        )


    def generate(self, z):
        mn = self.diagnose_month(z)
        return mn

    def call(self, x):
        mnth = self.generate(x[0])
        return mnth

    def compute_loss(self, x):
        mnth = self.generate(x[0])
        target = x[1]
        cce_MNTH = tf.keras.metrics.categorical_crossentropy(mnth, target)
        target = tf.roll(x[1], shift=1, axis=1)
        cce_MNTH += tf.keras.metrics.categorical_crossentropy(mnth, target) * 0.25
        target = tf.roll(x[1], shift=-1, axis=1)
        cce_MNTH += tf.keras.metrics.categorical_crossentropy(mnth, target) * 0.25
        cce_MNTH -= 16.11 * 0.5
        return cce_MNTH

    @tf.function
    def train_on_batch(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss_values = self.compute_loss(x)
        gradients = tape.gradient(loss_values, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
