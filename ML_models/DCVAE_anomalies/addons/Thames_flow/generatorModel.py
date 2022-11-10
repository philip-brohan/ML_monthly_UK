# Model to generate Thames flow from latent space vector

import tensorflow as tf


class GeneratorM(tf.keras.Model):

    # Initialiser - set up instance and define the models
    def __init__(self):
        super(GeneratorM, self).__init__()

        # Hyperparameters
        # Latent space dimension
        self.latent_dim = 100
        # Max gradient to apply in optimizer
        self.max_gradient = 2.0

        # Model to generate flow from latent space
        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(
                    units=100,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Dense(
                    units=100,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Dense(units=20, activation=None),
                tf.keras.layers.Softmax(),
            ]
        )

    # Call the generator model with a batch of points in latent space and return a
    #  batch of flows
    def call(self, z, training=True):
        generated = self.generator(z, training=training)
        return generated

    @tf.function
    def fit_loss(self, generated, target, climatology):

        # Metric is fractional variance reduction compared to climatology
        # skill = tf.reduce_mean(tf.math.squared_difference(generated, target))
        # guess = tf.reduce_mean(tf.math.squared_difference(climatology, target))
        skill = tf.reduce_mean(
            tf.keras.metrics.categorical_crossentropy(generated, target)
        )
        return skill
        # return skill / guess

    # Calculate the losses from autoencoding a batch of inputs
    # We are calculating a seperate loss for each variable, and for for the
    #  two components of the latent space KLD regularizer. This is useful
    #  for monitoring and debugging, but the weight update only depends
    #  on a single value (their sum).
    @tf.function
    def compute_loss(self, x, training):
        generated = self.call(x[0], training=training)
        clim = generated * 0.0 + tf.reduce_mean(x[1])  # Climatology
        loss = self.fit_loss(generated, x[1], clim)
        return loss

    # Run the generator for one batch, calculate the errors, calculate the
    #  gradients and update the layer weights.
    @tf.function
    def train_on_batch(self, x, optimizer):
        with tf.GradientTape() as tape:
            overall_loss = self.compute_loss(x, training=True)
        gradients = tape.gradient(overall_loss, self.trainable_variables)
        # Clip the gradients - helps against sudden numerical problems
        gradients = [tf.clip_by_norm(g, self.max_gradient) for g in gradients]
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
