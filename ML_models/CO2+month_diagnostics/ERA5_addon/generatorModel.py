# Specify a Deep Convolutional Generator
#  Maps encoded HadUK-Grid monthly fields to ERA5 reconstructions

import os
import sys
import tensorflow as tf
import numpy as np


# Hyperparameters
# Global error scale
RMSE_scale = 10000
# Relative importances of each variable in error
PRMSL_scale = 1.0
SST_scale = 1.0
T2M_scale = 1.0
PRATE_scale = 1.0


class DCG(tf.keras.Model):
    def __init__(self):
        super(DCG, self).__init__()
        self.latent_dim = 20

        self.fields_decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=45 * 28 * 40, activation=tf.nn.elu),
                tf.keras.layers.Reshape(target_shape=(45, 28, 40)),
                tf.keras.layers.Conv2DTranspose(
                    filters=20,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=10,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=5 * 2,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=5 * 4,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=4, kernel_size=3, strides=2, padding="same"
                ),
            ]
        )

    def decode(self, z):
        decoded = self.fields_decoder(z)
        return decoded

    def call(self, x):
        latent = self.reparameterize(x[0], x[1])
        decoded = self.decode(latent)
        return decoded

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean


def compute_loss(model, x):
    latent = model.reparameterize(x[0], x[1])
    encoded = model.decode(latent)
    rmse_PRMSL = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(encoded[:, :, :, 0], x[2][:, :, :, 0]),
            axis=[1],
        )
        * RMSE_scale
        * PRMSL_scale
    )
    rmse_SST = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(encoded[:, :, :, 1], x[2][:, :, :, 1]),
            axis=[1],
        )
        * RMSE_scale
        * SST_scale
    )
    rmse_T2M = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(encoded[:, :, :, 2], x[2][:, :, :, 2]),
            axis=[1],
        )
        * RMSE_scale
        * T2M_scale
    )
    rmse_PRATE = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(encoded[:, :, :, 3], x[2][:, :, :, 3]),
            axis=[1],
        )
        * RMSE_scale
        * PRATE_scale
    )
    return tf.stack([rmse_PRMSL, rmse_SST, rmse_T2M, rmse_PRATE,])


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        vstack = compute_loss(model, x)
        metric = tf.reduce_mean(tf.math.reduce_sum(vstack, axis=0))
    gradients = tape.gradient(metric, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
