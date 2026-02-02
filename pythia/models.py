# pylint:disable=too-many-lines, import-error, no-name-in-module
"""
This module defines the deep learning models used in PyThia,
which are built using `keras`.
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model, ops
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K


class Memoryless(Model):  # pylint: disable=abstract-method, too-many-ancestors
    """
    This model performs the following task: given two images `img0` and `img1`,
    it predicts the next image `img2`.

    It does so by combining two components: an encoder-decoder pair and a predictor.
    The encoder-decoder pair maps between images and latent space embeddings.
    The predictor takes the two embeddings of `img0` and `img1` and maps
    it to the predicted embedding of `img2`.

    The encoder-decoder pair in inspired by variational auto-encoders and the code
    herein is adapted from section 8.4. "Generating images with
    variational autoencoders" from Francois Chollet's book "Deep Learning with Python".

    It is called `Memoryless` because it predicts a future image, namely `img2`,
    without access to the past beyond the immediate past. This distinguishes it from
    the `LEMON` (Latent Embedding in Memory - Oracle Network) model below
    which stores information about past beyond the immediate past into memory.

    The main purpose of the `Memoryless` model is actually not to perform
    the prediction itself, but rather it is to serve as pre-training for
    the encoder-decoder pair. Indeed, since the architecture of the `predictor`
    component of the `Memoryless` model is particularly simple, it requires
    the encoder-decoder pair to learn features that are rich in information able
    to describe the dynamics at hand.
    """

    def __init__(
        self, img_shape=(32, 32, 1), latent_dim=12, kl_regularization_parameter=5e-4
    ):
        """
        Initialize an instance of the `Memoryless` model class.

        Arguments
        img_shape                       The shape of the images passed as inputs to
                                        the model and returned as output by the model.
                                        Expected to be `(width, height, 1)`
                                        (The last dimension being `1` means that
                                        the model expects grayscale images.)
        latent_dim                      The dimension of the latent space.
        kl_regularization_parameter     A regularization parameter appearing in front
                                        of Kullback-Leibler divergence terms in the loss
                                        function.
        """

        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.kl_regularization_parameter = kl_regularization_parameter

        # `shape_before_flattening` is initialized when building the encoder and
        # ensures that the output of the encoder is consistent with the input
        # of the decoder
        self._shape_before_flattening = None

        # Build the three components of the model
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.predictor = self._build_predictor()

    def _build_encoder(self) -> Model:
        """
        Build the encoder.

        The encoder is a convolutional neural network which maps images
        to latent space embeddings.

        Returns
        `tensorflow.keras.Model` object describing the encoder.
        """
        input_img = layers.Input(shape=self.img_shape)

        x = layers.Conv2D(32, 3, padding="same", activation="relu")(input_img)
        x = layers.Conv2D(64, 3, padding="same", activation="relu", strides=2)(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

        self._shape_before_flattening = x.shape[1:]

        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)

        # These are the mean and log-variance of the embedding
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        return Model(input_img, [z_mean, z_log_var], name="encoder")

    def _build_predictor(self) -> Model:
        """
        Build the predictor.

        The predictor is a neural network which maps the latent space embeddings
        of two images in the immediate past and maps them to the predicted latent space
        embedding of the image in the immediate future.

        It is intentionally chosen to be very simple (it is essentially a single linear
        regression layer) in order to ensure that the encoder-decoder pair learn
        features that have rich information that can be used to predict the next image.

        Returns
        `tensorflow.keras.Model` object describing the predictor.
        """
        predictor_input = layers.Input(shape=(4 * self.latent_dim,))
        predictor_output = layers.Dense(2 * self.latent_dim, activation="linear")(
            predictor_input
        )

        return Model(predictor_input, predictor_output, name="predictor")

    def _build_decoder(self) -> Model:
        """
        Build the decoder.

        The decoder is a convolutional neural network which maps latent space embeddings
        to images.

        Returns
        `tensorflow.keras.Model` object describing the decoder.
        """
        decoder_input = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense(np.prod(self._shape_before_flattening), activation="relu")(
            decoder_input
        )
        x = layers.Reshape(self._shape_before_flattening)(x)
        x = layers.Conv2DTranspose(32, 3, padding="same", activation="relu", strides=2)(
            x
        )
        x = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)

        return Model(decoder_input, x, name="decoder")

    def _sample(self, z_mean, z_log_var):
        """
        Sample from the distribution over the latent space
        using the reparametrization trick.

        See Chapter 19. "Autoencoders" from Bishop and Bishop's "Deep Learning:
        Foundations and Concepts" for details on the reparametrization trick.

        Arguments
        z_mean      Mean of the latent space distribution, which is assumed to be
                    a multivariate Gaussian.
        z_log_var   Logarithm of the variance of the latent space distribution,
                    which is assumed to be a multivariate Gaussian.

        Returns
        A vector in latent space sampled from the multivariate Gaussian distribution
        with mean `z_mean` and variance `exp(z_log_var)`.
        """
        batch_size = ops.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, self.latent_dim))
        return z_mean + ops.exp(z_log_var) * epsilon

    def encode(self, images):
        """
        Encode a batch of images using the encoder.

        Argument
        images      Array with shape `(n_samples, width, height, channels)`,
                    where typically `channels = 1`.

        Returns
        Array with shape `(n_samples, latent_dim)` encoding each of the input images
        into the latent space using the encoder.
        """
        return self.encoder(images)

    def decode(self, embedding_vector):
        """
        Decode a batch of embedding vectors into images using the decoder.

        Argument
        embedding_vector    Array with shape `(n_samples, latent_dim)` encoding a series
                            of samples into the latent space.

        Returns             Array with shape `(n_samples, width, height, channels)`,
                            where typically `channels = 1`, of the images decoded
                            from the embedding vectors passed as inputs.
        """
        return self.decoder(embedding_vector)

    def call(  # pylint:disable=too-many-locals, arguments-differ
        self,
        inputs,
        training=None,
    ):
        """
        Forward pass through the model, first through the encoder,
        then through the predictor, and finally through the decoder.

        Arguments
        inputs      Tuple `(img0, img1, img2)` where `img0` and `img1` are images from
                    the immediate past (from, say, `t = 0` and `t = 1` respectively) and
                    where `img2` is the image from the immediate future (say, `t = 2`)
                    to be predicted.
        training    Boolean indicating to built-in `keras` methods whether or not
                    the model is currently in training mode.

        Returns
        Tuple `(x0, x1, x2)` where `x0` and `x1` are the reconstructions of `img0` and
        `img1`, respectively, obtained by using solely the encoder and decoder (without
        appealing to the predictor) and where `x2` is the prediction of `img2`.

        Note that in practice the model is not directly called via this method, or via
        the built-in `keras` method `predict`. Instead, the function
        `predict_next_image` below is used in practice to predict `x2` from `img0` and
        `img1`.
        """

        img0, img1, img2 = inputs

        # Encode the first two images into latent space embeddings
        z_mean0, z_log_var0 = self.encoder(img0)
        z_mean1, z_log_var1 = self.encoder(img1)

        # Predict the latent space embedding of the image in the future
        predictor_input = ops.concatenate(
            [z_mean0, z_log_var0, z_mean1, z_log_var1], axis=-1
        )
        predictor_output = self.predictor(predictor_input)
        z_mean2 = predictor_output[:, : self.latent_dim]
        z_log_var2 = predictor_output[:, self.latent_dim :]

        # Sample from the distributions corresponding to the embedding vectors above
        # Note: The samples drawn from the embedding vectors corresponding to `img0` and
        # `img1` are not used in the prediction. Nonetheless they are computed since
        # they are used in the computation of the loss, and specifically
        # in the computation of the 'reconstruction error' contribution to the loss.
        z0 = self._sample(z_mean0, z_log_var0)
        z1 = self._sample(z_mean1, z_log_var1)
        z2 = self._sample(z_mean2, z_log_var2)

        # Decode the latent space vectors sampled above into images.
        # Note: as noted above, the reconstructed images `x0` and `x1` corresponding
        # to `img0` and `img1` are not used in the prediction of the future image `x2`.
        # They are, however, used in the computation of the 'reconstruction error'
        # component of the loss, which is why they are computed here.
        x0 = self.decoder(z0)
        x1 = self.decoder(z1)
        x2 = self.decoder(z2)

        # If the model is currently training, add the loss terms resulting fro
        # this sample.
        if training:
            self._add_losses(
                img0,
                img1,
                img2,
                x0,
                x1,
                x2,
                z_mean0,
                z_log_var0,
                z_mean1,
                z_log_var1,
                z_mean2,
                z_log_var2,
            )

        return x0, x1, x2

    def _loss(self, true_img, decoded_img, z_mean, z_log_var):
        """
        Compute the loss.

        The loss has two components: a reconstruction loss and
        a Kullback-Leibler regularization term.
        - The reconstruction loss is a sum of three binary cross-entropy terms,
          one for each of the three images and its reconstruction.
        - The Kullback-Leibler regularization term is the Kullback-Leibler
          regularization term used in variational autoencoders. This term essentially
          promotes the encoder-decoder pair learning robust, meaningful features.
          (See Francois Chollet's "Deep Learning with Python", section 8.4, or
          Bishop & Bishop's "Deep Learning: Foundations and Concepts", chapter 19,
          for more details.)

        Arguments
        true_img        The true image.
        decoded_img     The corresponding decoded image. The difference between this and
                        the true image could be due to pure reconstruction error
                        (i.e. passing through the encoder and then the decoder is not
                        the identity map) or due to both reconstruction and
                        prediction errors.
        z_mean          The mean of the multivariate Gaussian over the latent space.
        z_log_var       The logarithm of the variance of the Gaussian over the latent
                        space.

        Returns
        Total loss.
        """

        # Reconstruction loss
        true_img_flat = ops.reshape(true_img, (ops.shape(true_img)[0], -1))
        decoded_img_flat = ops.reshape(decoded_img, (ops.shape(decoded_img)[0], -1))
        reconstruction_loss = ops.mean(
            binary_crossentropy(true_img_flat, decoded_img_flat)
        )

        # Kullback-Leibler regularization term
        kl_loss = -self.kl_regularization_parameter * ops.mean(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1
        )

        return ops.mean(reconstruction_loss + kl_loss)

    def _add_losses(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        img0,
        img1,
        img2,
        x0,
        x1,
        x2,
        z_mean0,
        z_log_var0,
        z_mean1,
        z_log_var1,
        z_mean2,
        z_log_var2,
    ) -> None:
        """
        Combine together the three losses coming from the two images in the immediate
        past and the image in the immediate future.

        Arguments
        img0            The first image in the immediate past, corresponding to,
                        say, `t = 0`.
        img1            The second image in the immediate past, corresponding to,
                        say, `t = 1`.
        img2            The image in the immediate future, corresponding to,
                        say, `t = 2`.
        x0              The reconstructed image obtained from `img0`.
        x1              The reconstructed image obtained from `img0`.
        x2              The predicted image for `img2`.
        z_mean0         The mean of the multivariate Gaussian over the latent space
                        obtained by encoding `img0`.
        z_log_var0      The logarithm of the variance of the multivariate Gaussian
                        over the latent space obtained by encoding `img0`.
        z_mean1         The mean of the multivariate Gaussian over the latent space
                        obtained by encoding `img1`.
        z_log_var1      The logarithm of the variance of the multivariate Gaussian
                        over the latent space obtained by encoding `img1`.
        z_mean2         The mean of the multivariate Gaussian over the latent space
                        obtained by encoding `img2`.
        z_log_var2      The logarithm of the variance of the multivariate Gaussian
                        over the latent space obtained by encoding `img2`.

        This method does not return anything, instead it leverages the built-in
        `add_loss` method of `Model`.
        """
        loss0 = self._loss(img0, x0, z_mean0, z_log_var0)
        loss1 = self._loss(img1, x1, z_mean1, z_log_var1)
        loss2 = self._loss(img2, x2, z_mean2, z_log_var2)
        total_loss = loss0 + loss1 + loss2
        self.add_loss(total_loss)

    def predict_next_image(self, img0, img1):
        """
        Given two images in the immediate past, uses the full model (encoder, predictor,
        and decoder) to predict the next image in the immediate future.

        Arguments
        img0            The first image in the immediate past, corresponding to,
                        say, `t = 0`.
        img1            The second image in the immediate past, corresponding to,
                        say, `t = 1`.

        Returns
        x2              The prediction of the image in the immediate future,
                        corresponding to, say, `t = 2`.
        """

        # Encode the two images in the past into latent space embeddings
        z_mean0, z_log_var0 = self.encoder(img0)
        z_mean1, z_log_var1 = self.encoder(img1)

        # Predict the latent space embedding of the image in the future
        predictor_input = ops.concatenate(
            [z_mean0, z_log_var0, z_mean1, z_log_var1], axis=-1
        )
        predictor_output = self.predictor(predictor_input)
        z_mean2 = predictor_output[:, : self.latent_dim]
        z_log_var2 = predictor_output[:, self.latent_dim :]

        # Sample from the distribution encoding the image in the future and decode it
        z2 = self._sample(z_mean2, z_log_var2)
        x2 = self.decoder(z2)

        return x2


def build_model(  # pylint:disable=missing-function-docstring, too-many-locals, too-many-statements
    img_shape=(32, 32, 1), latent_dim=12
):

    class ScheduledSamplingPredictor(layers.Layer):
        """
        RNN predictor with scheduled sampling for curriculum learning.

        Gradually transitions from teacher forcing (using ground truth) to
        autoregressive prediction (using own outputs) during training.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.teacher_forcing_ratio = tf.Variable(
                1.0, trainable=False, dtype=tf.float32
            )
            self.lstm_cell = layers.LSTM(
                units=12 * latent_dim,
                return_sequences=False,
                return_state=True,
                activation="relu",
                kernel_initializer="he_normal",
            )
            self.output_layer = layers.Dense(units=2 * latent_dim, activation="linear")

        def build(self, input_shape):  # pylint:disable=missing-function-docstring
            # Build layers with known input shapes to avoid
            # runtime shape inference issues
            self.lstm_cell.build((None, 1, 2 * latent_dim))
            self.output_layer.build((None, 12 * latent_dim))
            super().build(input_shape)

        def compute_output_shape(
            self, input_shape
        ):  # pylint:disable=missing-function-docstring
            # Output: (batch, seq_len-1, 2*latent_dim)
            # We predict timesteps 2 to t from inputs 0 to t-1
            batch_size = input_shape[0][0]
            seq_len = input_shape[0][1]
            if seq_len is not None:
                return (batch_size, seq_len - 1, 2 * latent_dim)
            return (batch_size, None, 2 * latent_dim)

        def call(
            self, inputs, training=None
        ):  # pylint:disable=missing-function-docstring
            z_means, z_log_vars = inputs
            batch_size = tf.shape(z_means)[0]
            seq_len = tf.shape(z_means)[1]

            # Initialize LSTM hidden and cell states
            h_state = tf.zeros((batch_size, 12 * latent_dim), dtype=tf.float32)
            c_state = tf.zeros((batch_size, 12 * latent_dim), dtype=tf.float32)

            # TensorArray to collect predictions
            predictions = tf.TensorArray(
                dtype=tf.float32,
                size=seq_len - 1,
                dynamic_size=False,
                clear_after_read=False,
            )

            # Start with timestep 0 as input
            current_input = tf.concat([z_means[:, 0], z_log_vars[:, 0]], axis=-1)
            current_input = tf.expand_dims(
                current_input, 1
            )  # Shape: (batch, 1, 2*latent_dim)

            def loop_body(t, current_input, h_state, c_state, predictions):
                # Get prediction for timestep t+1 using input from timestep t-1
                rnn_output, h_state_new, c_state_new = self.lstm_cell(
                    current_input, initial_state=[h_state, c_state], training=training
                )
                pred = self.output_layer(rnn_output)  # Shape: (batch, 2*latent_dim)
                predictions = predictions.write(t - 1, pred)

                # Ground truth at timestep t
                ground_truth = tf.concat([z_means[:, t], z_log_vars[:, t]], axis=-1)

                # Scheduled sampling: choose input for next iteration
                if training:
                    # Randomly use ground truth or prediction based on current ratio
                    use_ground_truth = (
                        tf.random.uniform([], dtype=tf.float32)
                        < self.teacher_forcing_ratio
                    )
                    next_input = tf.cond(
                        use_ground_truth, lambda: ground_truth, lambda: pred
                    )
                else:
                    # During inference, always use model's predictions (autoregressive)
                    next_input = pred

                next_input = tf.expand_dims(next_input, 1)
                return t + 1, next_input, h_state_new, c_state_new, predictions

            def loop_cond(t, *_):
                return t < seq_len

            # Execute loop using tf.while_loop for graph compatibility
            _, _, _, _, predictions = tf.while_loop(
                loop_cond,
                loop_body,
                [tf.constant(1), current_input, h_state, c_state, predictions],
                parallel_iterations=1,  # Sequential execution required for RNN
                maximum_iterations=1000,  # Set max for XLA compilation
            )

            # Stack and transpose:
            # (seq_len-1, batch, 2*latent_dim) -> (batch, seq_len-1, 2*latent_dim)
            return tf.transpose(predictions.stack(), [1, 0, 2])

    class CustomVariationalLayer(  # pylint:disable=missing-class-docstring, too-few-public-methods
        layers.Layer
    ):
        def call(
            self, inputs
        ):  # pylint:disable=missing-function-docstring, too-many-locals
            (
                img_sequence,
                img_tilde,
                img_hat,
                z_means,
                z_log_vars,
                z_means_hat,
                z_log_vars_hat,
            ) = inputs

            # Reconstruction error
            img_for_recon = img_sequence[:, :-1, :, :, :]
            img_for_recon_flat = K.flatten(img_for_recon)
            img_tilde_flat = K.flatten(img_tilde)
            reconstruction_loss = K.mean(
                binary_crossentropy(img_for_recon_flat, img_tilde_flat)
            )

            # Prediction error
            img_for_pred = img_sequence[:, 2:, :, :, :]
            img_for_pred_flat = K.flatten(img_for_pred)
            img_hat_flat = K.flatten(img_hat)
            prediction_loss = K.mean(
                binary_crossentropy(img_for_pred_flat, img_hat_flat)
            )

            # KL divergences
            kl_pre = -5e-5 * K.mean(
                K.mean(1 + z_log_vars - K.square(z_means) - K.exp(z_log_vars), axis=-1)
            )
            kl_post = -5e-5 * K.mean(
                K.mean(
                    1 + z_log_vars_hat - K.square(z_means_hat) - K.exp(z_log_vars_hat),
                    axis=-1,
                )
            )

            total_loss = reconstruction_loss + prediction_loss + kl_pre + kl_post
            self.add_loss(total_loss)
            return img_sequence[:, 0:1, :, :, :]

    # Encoder (images to latent space)
    input_img = layers.Input(shape=img_shape)
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(
        input_img
    )
    x = layers.Conv2D(
        filters=64, kernel_size=3, padding="same", activation="relu", strides=(2, 2)
    )(x)
    x = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=32, activation="relu")(x)
    z_mean = layers.Dense(units=latent_dim)(x)
    z_log_var = layers.Dense(units=latent_dim)(x)
    encoder = Model(input_img, [z_mean, z_log_var])

    # Latent space sampling
    def sampling(args):
        z_mean, z_log_var = args
        shape = K.shape(z_mean)
        epsilon = K.random_normal(shape=shape, mean=0.0, stddev=1.0)
        return z_mean + K.exp(z_log_var) * epsilon

    # Decoder (latent space to images)
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation="relu")(
        decoder_input
    )
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.Conv2DTranspose(
        filters=32, kernel_size=3, padding="same", activation="relu", strides=(2, 2)
    )(x)
    x = layers.Conv2DTranspose(
        filters=1, kernel_size=3, padding="same", activation="sigmoid"
    )(x)
    decoder = Model(decoder_input, x)

    # Build the full model with variable-length image sequences
    img_sequence = layers.Input(shape=(None,) + img_shape)

    # Encode images at timesteps 0 to t-1
    img_for_encoding = layers.Lambda(lambda x: x[:, :-1, :, :, :])(img_sequence)

    def encode_sequence(img_seq):
        batch_size = K.shape(img_seq)[0]
        seq_len = K.shape(img_seq)[1]
        img_reshaped = K.reshape(img_seq, (-1,) + img_shape)
        z_mean_flat, z_log_var_flat = encoder(img_reshaped)
        z_mean_seq = K.reshape(z_mean_flat, (batch_size, seq_len, latent_dim))
        z_log_var_seq = K.reshape(z_log_var_flat, (batch_size, seq_len, latent_dim))
        return z_mean_seq, z_log_var_seq

    z_means, z_log_vars = layers.Lambda(encode_sequence)(img_for_encoding)

    scheduled_predictor = ScheduledSamplingPredictor(name="scheduled_predictor")
    predictor_output = scheduled_predictor([z_means, z_log_vars])

    z_means_hat = layers.Lambda(lambda x: x[:, :, :latent_dim])(predictor_output)
    z_log_vars_hat = layers.Lambda(lambda x: x[:, :, latent_dim:])(predictor_output)

    # Sample from encoded latent distributions
    z_samples_encoded = layers.Lambda(sampling)([z_means, z_log_vars])

    def decode_sequence(z_seq):
        batch_size = K.shape(z_seq)[0]
        seq_len = K.shape(z_seq)[1]
        z_flat = K.reshape(z_seq, (-1, latent_dim))
        x_flat = decoder(z_flat)
        x_seq = K.reshape(x_flat, (batch_size, seq_len) + img_shape)
        return x_seq

    img_tilde = layers.Lambda(decode_sequence)(z_samples_encoded)

    # Sample from predicted latent distributions
    z_samples_predicted = layers.Lambda(sampling)([z_means_hat, z_log_vars_hat])
    img_hat = layers.Lambda(decode_sequence)(z_samples_predicted)

    y = CustomVariationalLayer()(
        [
            img_sequence,
            img_tilde,
            img_hat,
            z_means,
            z_log_vars,
            z_means_hat,
            z_log_vars_hat,
        ]
    )

    # Compile the VAE model
    vae = Model(img_sequence, y)
    return vae, encoder, scheduled_predictor, decoder


class ScheduledSamplingCallback(Callback):
    """
    Gradually reduces teacher forcing ratio during training.

    Args:
        initial_ratio: Starting ratio (1.0 = always use ground truth)
        decay_rate: Multiplicative decay factor (e.g., 0.95)
        decay_every: Decay every N epochs
        min_ratio: Minimum ratio to maintain (default 0.0)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        predictor,
        initial_ratio=1.0,
        decay_rate=0.95,
        decay_every=5,
        min_ratio=0.0,
    ):
        super().__init__()
        self.predictor = predictor
        self.initial_ratio = initial_ratio
        self.decay_rate = decay_rate
        self.decay_every = decay_every
        self.min_ratio = min_ratio

    def on_train_begin(
        self, logs=None
    ):  # pylint:disable=unused-argument, disable=missing-function-docstring
        # Ensure we start at initial ratio
        self.predictor.teacher_forcing_ratio.assign(self.initial_ratio)
        print(f"\nStarting with teacher forcing ratio: {self.initial_ratio:.4f}")

    def on_epoch_end(
        self, epoch, logs=None
    ):  # pylint:disable=unused-argument, disable=missing-function-docstring
        if (epoch + 1) % self.decay_every == 0:
            current_ratio = self.predictor.teacher_forcing_ratio.numpy()
            new_ratio = max(self.min_ratio, current_ratio * self.decay_rate)
            self.predictor.teacher_forcing_ratio.assign(new_ratio)
            print(f"\nEpoch {epoch+1}: Teacher forcing ratio = {new_ratio:.4f}")
