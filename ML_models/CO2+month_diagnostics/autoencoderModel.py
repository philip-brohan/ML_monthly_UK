# Specify a Deep Convolutional Variational AutoEncoder
#  for the HadUK-Grid monthly fields - with CO2 and month as scalar diagnostics

import os
import sys
import tensorflow as tf
import numpy as np


# Hyperparameters
# Ratio of RMSE to KLD in error
RMSE_scale = 10000
# Relative importances of each variable in error
PRMSL_scale = 1.0
SST_scale = 1.0
T2M_scale = 1.0
PRATE_scale = 1.0
CO2_scale = 1.0
MONTH_scale = 1.0


class DCVAE(tf.keras.Model):
    def __init__(self):
        super(DCVAE, self).__init__()
        self.latent_dim = 20
        self.fields_encoder = tf.keras.Sequential(
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
            ]
        )

        self.merge_to_latent = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(45 * 28 * 40 + 2)),
                tf.keras.layers.Dense(units=self.latent_dim * 2, activation=tf.nn.elu),
                tf.keras.layers.Dense(units=self.latent_dim * 2, activation=None),
            ]
        )

        self.fields_decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=45 * 28 * 40, activation=tf.nn.relu),
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

        self.diagnose_co2 = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=self.latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=1, activation=tf.nn.relu),
            ]
        )

        self.diagnose_month = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=self.latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=1, activation=tf.nn.relu),
            ]
        )

    def encode(self, x):
        field = x[0]
        c2 = x[1]
        mn = x[2]
        # Encode the field
        encf = self.fields_encoder(field)
        # Add the CO2 and month to the encoded state
        encf = tf.concat(
            [encf, tf.expand_dims(c2, axis=1), tf.expand_dims(mn, axis=1)], axis=1
        )
        # Convert the merged encoded state into a latent space
        mean, logvar = tf.split(
            self.merge_to_latent(encf), num_or_size_splits=2, axis=1
        )
        return mean, logvar

    def decode(self, z):
        decoded = self.fields_decoder(z)
        c2 = self.diagnose_co2(z)
        mn = self.diagnose_month(z)
        return (decoded, c2, mn)

    def call(self, x):
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        encoded = self.decode(latent)
        return encoded

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    # Decode each member of the batch several times, to make a sample
    #  returns a 4d tensor (size, batch, y, x)
    def sample_decode(self, mean, logvar, size=100):
        encoded = []
        eps = tf.random.normal(shape=(size, self.latent_dim))
        mean = tf.unstack(mean, axis=0)
        logvar = tf.unstack(logvar, axis=0)
        for batchI in range(len(mean)):
            latent = eps * tf.exp(logvar[batchI] * 0.5) + mean[batchI]
            encoded.append(self.decode(latent))
        return tf.stack(encoded, axis=0)

    # Autoencode each member of the batch several times, to make a sample
    def sample_call(self, x, size=100):
        mean, logvar = self.encode(x)
        encoded = self.sample_decode(mean, logvar, size=size)
        return encoded


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis,
    )


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    latent = model.reparameterize(mean, logvar)
    encoded = model.decode(latent)
    rmse_PRMSL = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                encoded[0][:, :, :, 0], x[0][:, :, :, 0]
            ),
            axis=[1],
        )
        * RMSE_scale
        * PRMSL_scale
    )
    rmse_SST = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                encoded[0][:, :, :, 1], x[0][:, :, :, 1]
            ),
            axis=[1],
        )
        * RMSE_scale
        * SST_scale
    )
    rmse_T2M = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                encoded[0][:, :, :, 2], x[0][:, :, :, 2]
            ),
            axis=[1],
        )
        * RMSE_scale
        * T2M_scale
    )
    rmse_PRATE = (
        tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                encoded[0][:, :, :, 3], x[0][:, :, :, 3]
            ),
            axis=[1],
        )
        * RMSE_scale
        * PRATE_scale
    )
    logpz = log_normal_pdf(latent, 0.0, 0.0) * -1
    logqz_x = log_normal_pdf(latent, mean, logvar)
    rmse_CO2 = (
        tf.keras.metrics.mean_squared_error(encoded[1], x[1]) * RMSE_scale * CO2_scale
    )
    rmse_MONTH = (
        tf.keras.metrics.mean_squared_error(encoded[2], x[2]) * RMSE_scale * MONTH_scale
    )
    return tf.stack(
        [
            rmse_PRMSL,
            rmse_SST,
            rmse_T2M,
            rmse_PRATE,
            logpz,
            logqz_x,
            rmse_CO2,
            rmse_MONTH,
        ]
    )


@tf.function  # Optimiser ~25% speedup on VDI (CPU-only)
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
