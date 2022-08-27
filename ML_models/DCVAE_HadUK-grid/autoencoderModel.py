# Specify a Deep Convolutional Variational AutoEncoder
#  for the HadUK-Grid monthly fields.


import tensorflow as tf


class DCVAE(tf.keras.Model):

    # Initialiser - set up instance and define the models
    def __init__(self):
        super(DCVAE, self).__init__()

        # Hyperparameters
        # Latent space dimension
        self.latent_dim = 20
        # Ratio of RMSE to KLD in error
        self.RMSE_scale = tf.constant(10000.0, dtype=tf.float32)
        # Relative importances of each variable in error
        self.PRMSL_scale = tf.constant(1.0, dtype=tf.float32)
        self.SST_scale = tf.constant(1.0, dtype=tf.float32)
        self.T2M_scale = tf.constant(1.0, dtype=tf.float32)
        self.PRATE_scale = tf.constant(1.0, dtype=tf.float32)

        # Measures for performance on current batch
        self.rmse_PRMSL = tf.Variable(0.0)
        self.rmse_PRATE = tf.Variable(0.0)
        self.rmse_T2M = tf.Variable(0.0)
        self.rmse_SST = tf.Variable(0.0)
        self.logpz = tf.Variable(0.0)
        self.logqz_x = tf.Variable(0.0)
        self.loss = tf.Variable(0.0)

        # Model to encode input to latent space distribution
        self.encoder = tf.keras.Sequential(
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
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        # Model to generate output from latent space
        self.generator = tf.keras.Sequential(
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

    # Call the encoder model with a batch of input examples and return a batch of
    #  means and a batch of variances of the encoded latent space PDFs.
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    # Sample a batch of points in latent space from the encoded means and variances
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    # Call the generator model with a batch of points in latent space and return a
    #  batch of outputs
    def generate(self, z):
        generated = self.generator(z)
        return generated

    # Run the full VAE - convert a batch of inputs to one of outputs
    # Note, does not do training - there's another function for that.
    def call(self, x):
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        generated = self.generate(latent)
        return generated

    # Utility function to calculte fit of sample to N(mean,logvar)
    # Used in loss calculation
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * 3.141592653589793)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    # Calculate the losses from autoencoding a batch of inputs
    # We are calculating a seperate loss for each variable, and for for the
    #  two components of the latent space KLD regularizer. This is useful
    #  for monitoring and debugging, but the weight update only depends
    #  on a single value (their sum).
    # @tf.function - Breaks function (all losses=0, why?)
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        generated = self.generate(latent)

        self.rmse_PRMSL = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(generated[:, :, :, 0], x[:, :, :, 0])
                )
            )
            * self.RMSE_scale
            * self.PRMSL_scale
        )
        self.rmse_SST = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(generated[:, :, :, 1], x[:, :, :, 1])
                )
            )
            * self.RMSE_scale
            * self.SST_scale
        )
        self.rmse_T2M = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(generated[:, :, :, 2], x[:, :, :, 2])
                )
            )
            * self.RMSE_scale
            * self.T2M_scale
        )
        self.rmse_PRATE = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(generated[:, :, :, 3], x[:, :, :, 3])
                )
            )
            * self.RMSE_scale
            * self.PRATE_scale
        )
        self.logpz = self.log_normal_pdf(latent, 0.0, 0.0) * -1
        self.logqz_x = self.log_normal_pdf(latent, mean, logvar)
        self.loss = (
            self.rmse_PRMSL
            + self.rmse_SST
            + self.rmse_T2M
            + self.rmse_PRATE
            + self.logpz
            + self.logqz_x
        )

        return self.loss

    # Run the autoencoder for one batch, calculate the errors, calculate the
    #  gradients and update the layer weights.
    @tf.function
    def train_on_batch(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss_value = self.compute_loss(x)
        gradients = tape.gradient(loss_value, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
