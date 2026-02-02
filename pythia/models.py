# pylint:disable=too-many-lines
"""
This module defines the deep learning models used in PyThia,
which are built using `keras`.
"""

import numpy as np
import tensorflow as tf

from keras import layers, Model, ops
from keras.callbacks import Callback
from keras.random import normal
from keras.losses import binary_crossentropy


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
        epsilon = normal(shape=(batch_size, self.latent_dim))
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


class ScheduledSamplingPredictor(  # pylint: disable=abstract-method, too-many-ancestors
    layers.Layer
):
    """
    RNN predictor with scheduled sampling for curriculum learning.

    This layer implements an LSTM-based predictor that learns to forecast future
    latent space embeddings in a sequence. It uses scheduled sampling, a curriculum
    learning technique that gradually transitions from teacher forcing (where the
    model is fed ground truth values during training) to autoregressive prediction
    (where the model uses its own predictions as inputs).

    The transition is controlled by `teacher_forcing_ratio`, which starts at 1.0
    (100% teacher forcing) and is gradually decayed during training. At each training
    step, the model randomly decides whether to use the ground truth or its own
    prediction based on this ratio. During inference, the model always uses its own
    predictions (autoregressive mode).

    This approach helps the model learn more robust predictions by gradually
    exposing it to its own errors during training, rather than experiencing a
    sudden shift from perfect inputs to imperfect ones at inference time.
    """

    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.teacher_forcing_ratio = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.lstm_cell = layers.LSTM(
            units=12 * latent_dim,
            return_sequences=False,
            return_state=True,
            activation="relu",
            kernel_initializer="he_normal",
        )
        self.output_layer = layers.Dense(units=2 * latent_dim, activation="linear")

    def build(self, input_shape):
        # Build layers with known input shapes
        self.lstm_cell.build((None, 1, 2 * self.latent_dim))
        self.output_layer.build((None, 12 * self.latent_dim))
        super().build(input_shape)

    def call(self, inputs, training=None):  # pylint:disable=arguments-differ
        z_means, z_log_vars = inputs
        batch_size = tf.shape(z_means)[0]
        seq_len = tf.shape(z_means)[1]

        h_state = tf.zeros((batch_size, 12 * self.latent_dim), dtype=tf.float32)
        c_state = tf.zeros((batch_size, 12 * self.latent_dim), dtype=tf.float32)

        predictions = tf.TensorArray(
            dtype=tf.float32,
            size=seq_len - 1,
            dynamic_size=False,
            clear_after_read=False,
        )

        # Warm up with timestep 0 → predict timestep 1 (not saved)
        current_input = tf.concat([z_means[:, 0], z_log_vars[:, 0]], axis=-1)
        current_input = tf.expand_dims(current_input, 1)
        rnn_output, h_state, c_state = self.lstm_cell(
            current_input, initial_state=[h_state, c_state], training=training
        )
        pred = self.output_layer(rnn_output)

        def loop_body(t, current_input, h_state, c_state, predictions):
            # Use current_input to predict t+1
            rnn_output, h_state_new, c_state_new = self.lstm_cell(
                current_input, initial_state=[h_state, c_state], training=training
            )
            pred = self.output_layer(rnn_output)
            # Save prediction (for timestep t+1, stored at index t-1)
            predictions = predictions.write(t - 1, pred)

            # Ground truth at timestep t for scheduled sampling
            ground_truth = tf.concat([z_means[:, t], z_log_vars[:, t]], axis=-1)

            if training:
                use_ground_truth = (
                    tf.random.uniform([], dtype=tf.float32) < self.teacher_forcing_ratio
                )
                next_input = tf.cond(
                    use_ground_truth, lambda: ground_truth, lambda: pred
                )
            else:
                next_input = pred

            next_input = tf.expand_dims(next_input, 1)
            return [t + 1, next_input, h_state_new, c_state_new, predictions]

        def loop_cond(t, *_):
            return t < seq_len

        # Start loop at t=1, using pred from timestep 0
        _, _, _, _, predictions = tf.while_loop(
            loop_cond,
            loop_body,
            [tf.constant(1), tf.expand_dims(pred, 1), h_state, c_state, predictions],
            parallel_iterations=1,
            maximum_iterations=1000,
        )

        return tf.transpose(predictions.stack(), [1, 0, 2])


class LEMON(Model):  # pylint: disable=abstract-method, too-many-ancestors
    """
    This model performs the following task: given a sequence of images, it learns
    to predict future images in the sequence.

    It does so by combining three components: an encoder, a decoder, and a predictor.
    The encoder-decoder pair maps between images and latent space embeddings, inspired
    by variational autoencoders. The predictor is an LSTM-based network that takes
    the sequence of latent embeddings and predicts future latent embeddings, which
    are then decoded back into images.

    The predictor uses scheduled sampling, a curriculum learning technique that
    gradually transitions from teacher forcing (using ground truth latent codes during
    training) to autoregressive prediction (using the model's own predictions). This
    helps the model learn more robust predictions by gradually exposing it to its
    own errors during training.

    The encoder-decoder architecture is adapted from section 8.4 "Generating images with
    variational autoencoders" from Francois Chollet's book "Deep Learning with Python".
    The scheduled sampling technique helps bridge the gap between training (with
    perfect inputs) and inference (with the model's own predictions).

    During training, the model optimizes four loss components:
    1. Reconstruction loss: How well the encoder-decoder pair can reconstruct
       the input images (for timesteps 0 to t-1).
    2. Prediction loss: How well the full model (encoder, predictor, decoder)
       can predict future images (for timesteps 2 to t).
    3. KL divergence (pre-prediction): Regularization term for the encoded latent
       distributions, encouraging them to be close to a standard normal distribution.
    4. KL divergence (post-prediction): Regularization term for the predicted latent
       distributions, similarly encouraging normality.

    These loss components together ensure that the latent space is well-structured
    and that the model learns meaningful features for both reconstruction and
    prediction.
    """

    def __init__(
        self, img_shape=(32, 32, 1), latent_dim=12, kl_regularization_parameter=5e-5
    ):
        """
        Initialize an instance of the `SequenceVAE` model class.

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

        # Build components
        self._shape_before_flattening = None
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

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        return Model(input_img, [z_mean, z_log_var], name="encoder")

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

    def _build_predictor(self) -> ScheduledSamplingPredictor:
        """
        Build the predictor.

        The predictor is an LSTM-based neural network with scheduled sampling that
        takes a sequence of latent space embeddings and predicts future latent space
        embeddings. It uses curriculum learning to gradually transition from teacher
        forcing (using ground truth) to autoregressive prediction
        (using its own outputs).

        Returns
        `ScheduledSamplingPredictor` object describing the predictor.
        """
        return ScheduledSamplingPredictor(self.latent_dim, name="predictor")

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
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(z_log_var) * epsilon

    def encode_sequence(self, img_sequence):
        """
        Encode a sequence of images into latent space embeddings.

        This method processes an entire sequence of images through the encoder
        by flattening the sequence dimension, encoding all images in parallel,
        and then reshaping back to sequence format.

        Arguments
        img_sequence    Array with shape
                        `(batch_size, seq_len, width, height, channels)`,
                        where typically `channels = 1`.

        Returns
        Tuple `(z_mean_seq, z_log_var_seq)` where both arrays have shape
        `(batch_size, seq_len, latent_dim)`, representing the mean and log-variance
        of the latent space distributions for each image in the sequence.
        """
        # Get dynamic shapes
        shape = tf.shape(img_sequence)
        batch_size = shape[0]
        seq_len = shape[1]

        # Flatten sequence for batch encoding
        img_flat = tf.reshape(img_sequence, (-1,) + self.img_shape)
        z_mean_flat, z_log_var_flat = self.encoder(img_flat)

        # Reshape back to sequence
        z_mean_seq = tf.reshape(z_mean_flat, (batch_size, seq_len, self.latent_dim))
        z_log_var_seq = tf.reshape(
            z_log_var_flat, (batch_size, seq_len, self.latent_dim)
        )

        return z_mean_seq, z_log_var_seq

    def decode_sequence(self, z_sequence):
        """
        Decode a sequence of latent codes into images.

        This method processes an entire sequence of latent space embeddings through
        the decoder by flattening the sequence dimension, decoding all embeddings
        in parallel, and then reshaping back to sequence format.

        Arguments
        z_sequence      Array with shape `(batch_size, seq_len, latent_dim)` encoding
                        a sequence of samples in the latent space.

        Returns
        Array with shape `(batch_size, seq_len, width, height, channels)`,
        where typically `channels = 1`, of the images decoded from the
        embedding vectors passed as inputs.
        """
        batch_size = tf.shape(z_sequence)[0]
        seq_len = tf.shape(z_sequence)[1]

        # Flatten sequence for batch decoding
        z_flat = tf.reshape(z_sequence, (-1, self.latent_dim))
        img_flat = self.decoder(z_flat)

        # Reshape back to sequence
        img_sequence = tf.reshape(img_flat, (batch_size, seq_len) + self.img_shape)

        return img_sequence

    def call(self, inputs, training=None):  # pylint:disable=arguments-differ
        """
        Forward pass through the model, first through the encoder to obtain latent
        embeddings of the input sequence, then through the predictor to forecast
        future latent embeddings, and finally through the decoder to reconstruct
        and predict images.

        The forward pass proceeds as follows:
        1. Encode images at timesteps 0 to t-1 into latent space embeddings
           (mean and log-variance).
        2. Pass the encoded sequence through the predictor to obtain predicted
           latent embeddings for timesteps 2 to t (note the offset: we use inputs
           from 0 to t-1 to predict outputs from 2 to t, giving us one-step-ahead
           predictions after the first timestep).
        3. Sample from both the encoded and predicted latent distributions using
           the reparametrization trick.
        4. Decode both sets of samples to obtain reconstructed images (for timesteps
           0 to t-1) and predicted images (for timesteps 2 to t).

        During training, this method also computes and adds four loss components:
        reconstruction loss, prediction loss, and two KL divergence regularization
        terms (see `_add_losses` for details).

        Arguments
        inputs      Image sequence with shape
                    `(batch_size, seq_len, width, height, channels)`.

                    The model uses timesteps 0 to t-1 as inputs for reconstruction and
                    prediction, and timesteps 2 to t as targets for prediction.
        training    Boolean indicating to built-in `keras` methods whether or not
                    the model is currently in training mode.

        Returns
        Array with shape `(batch_size, 1, width, height, channels)` containing the
        first image of the input sequence. This is a dummy output required by the
        `keras` API, as the actual predictions are computed internally and losses
        are added via `self.add_loss()`.

        Note that in practice the model is not directly called via this method, or via
        the built-in `keras` method `predict`. Instead, the function `predict_future`
        is used to generate future predictions autoregressively.
        """
        img_sequence = inputs

        # Encode images at timesteps 0 to t-1
        img_for_encoding = img_sequence[:, :-1, :, :, :]
        z_means, z_log_vars = self.encode_sequence(img_for_encoding)

        # Predict latent codes for timesteps 2 to t
        predictor_output = self.predictor([z_means, z_log_vars], training=training)
        z_means_hat = predictor_output[:, :, : self.latent_dim]
        z_log_vars_hat = predictor_output[:, :, self.latent_dim :]

        # Sample and decode for reconstruction
        z_samples_encoded = self._sample(z_means, z_log_vars)
        img_tilde = self.decode_sequence(z_samples_encoded)

        # Sample and decode for prediction
        z_samples_predicted = self._sample(z_means_hat, z_log_vars_hat)
        img_hat = self.decode_sequence(z_samples_predicted)

        if training:
            self._add_losses(
                img_sequence,
                img_tilde,
                img_hat,
                z_means,
                z_log_vars,
                z_means_hat,
                z_log_vars_hat,
            )

        # Return dummy output
        return img_sequence[:, 0:1, :, :, :]

    def _add_losses(  # pylint:disable=too-many-arguments, too-many-locals
        self,
        img_sequence,
        img_tilde,
        img_hat,
        z_means,
        z_log_vars,
        z_means_hat,
        z_log_vars_hat,
    ):
        """
        Compute and combine the loss components for training.

        The loss has four components:

        1. Reconstruction loss: A binary cross-entropy term measuring how well
           the encoder-decoder pair can reconstruct the input images at timesteps
           0 to t-1. This ensures the encoder-decoder learns to preserve image
           information through the latent space bottleneck.

        2. Prediction loss: A binary cross-entropy term measuring how well the
           full model (encoder, predictor, decoder) can predict future images at
           timesteps 2 to t given the sequence up to timestep t-1. This is the
           primary predictive objective of the model.

        3. KL divergence (pre-prediction): The Kullback-Leibler divergence between
           the encoded latent distributions and a standard normal distribution.
           This regularization term, used in variational autoencoders, encourages
           the latent space to be well-structured and the learned features to be
           robust and meaningful.

        4. KL divergence (post-prediction): The Kullback-Leibler divergence between
           the predicted latent distributions and a standard normal distribution.
           This similarly regularizes the predicted latent codes to maintain the
           same distributional properties as the encoded ones.

        See Francois Chollet's "Deep Learning with Python", section 8.4, or
        Bishop & Bishop's "Deep Learning: Foundations and Concepts", chapter 19,
        for more details on the role of KL divergence in variational autoencoders.

        Arguments
        img_sequence    The full input image sequence with shape
                        `(batch_size, seq_len, width, height, channels)`.
        img_tilde       Reconstructed images from timesteps 0 to t-1, obtained by
                        encoding and then decoding the input images.
        img_hat         Predicted images for timesteps 2 to t, obtained by encoding
                        the input sequence, using the predictor to forecast future
                        latent codes, and decoding those predictions.
        z_means         Mean of the multivariate Gaussian over the latent space
                        obtained by encoding timesteps 0 to t-1.
        z_log_vars      Logarithm of the variance of the multivariate Gaussian
                        over the latent space obtained by encoding timesteps 0 to t-1.
        z_means_hat     Mean of the multivariate Gaussian over the latent space
                        predicted for timesteps 2 to t.
        z_log_vars_hat  Logarithm of the variance of the multivariate Gaussian
                        over the latent space predicted for timesteps 2 to t.

        This method does not return anything, instead it leverages the built-in
        `add_loss` method of `Model`.
        """
        # Reconstruction loss
        img_for_recon = img_sequence[:, :-1, :, :, :]
        img_for_recon_flat = tf.reshape(img_for_recon, (tf.shape(img_for_recon)[0], -1))
        img_tilde_flat = tf.reshape(img_tilde, (tf.shape(img_tilde)[0], -1))
        reconstruction_loss = tf.reduce_mean(
            binary_crossentropy(img_for_recon_flat, img_tilde_flat)
        )

        # Prediction loss
        img_for_pred = img_sequence[:, 2:, :, :, :]
        img_for_pred_flat = tf.reshape(img_for_pred, (tf.shape(img_for_pred)[0], -1))
        img_hat_flat = tf.reshape(img_hat, (tf.shape(img_hat)[0], -1))
        prediction_loss = tf.reduce_mean(
            binary_crossentropy(img_for_pred_flat, img_hat_flat)
        )

        # KL divergence for encoded latent codes
        kl_pre = -self.kl_regularization_parameter * tf.reduce_mean(
            tf.reduce_mean(
                1 + z_log_vars - tf.square(z_means) - tf.exp(z_log_vars), axis=-1
            )
        )

        # KL divergence for predicted latent codes
        kl_post = -self.kl_regularization_parameter * tf.reduce_mean(
            tf.reduce_mean(
                1 + z_log_vars_hat - tf.square(z_means_hat) - tf.exp(z_log_vars_hat),
                axis=-1,
            )
        )

        total_loss = reconstruction_loss + prediction_loss + kl_pre + kl_post
        self.add_loss(total_loss)

    def predict_future(
        self, initial_sequence, num_future_steps
    ):  # pylint: disable=too-many-locals
        """
        Given an initial sequence of images, uses the full model (encoder, predictor,
        and decoder) to autoregressively predict future images.

        This function first encodes the initial sequence to obtain latent embeddings,
        then uses these to initialize and warm up the LSTM state of the predictor by
        feeding the entire initial sequence through it. After warming up, it generates
        future predictions autoregressively: at each step, the predictor forecasts the
        next latent code, which is sampled and decoded into an image, and this predicted
        latent code is then used as input for the next prediction step.

        Arguments
        initial_sequence    Array with shape `(batch_size, t, width, height, channels)`
                            containing the initial sequence of images. The model will
                            use this sequence to initialize its internal state before
                            generating predictions.
        num_future_steps    Number of future timesteps to predict beyond the initial
                            sequence.

        Returns
        Array with shape `(batch_size, num_future_steps, width, height, channels)`
        containing the predicted future images.
        """
        batch_size = tf.shape(initial_sequence)[0]
        latent_dim = self.latent_dim

        # Encode initial sequence
        z_mean, z_log_var = self.encode_sequence(initial_sequence)
        seq_len = tf.shape(z_mean)[1]

        # Initialize LSTM state by feeding through initial sequence
        h_state = tf.zeros((batch_size, 12 * latent_dim), dtype=tf.float32)
        c_state = tf.zeros((batch_size, 12 * latent_dim), dtype=tf.float32)

        # Warm up LSTM state AND get first prediction
        current_input = None
        for t in range(seq_len):
            current_input = tf.concat([z_mean[:, t], z_log_var[:, t]], axis=-1)
            current_input = tf.expand_dims(current_input, 1)
            rnn_output, h_state, c_state = self.predictor.lstm_cell(
                current_input, initial_state=[h_state, c_state], training=False
            )

        # Get the first prediction from the warmed-up state
        pred = self.predictor.output_layer(rnn_output)

        # Generate future predictions autoregressively
        future_predictions = []
        current_input = tf.expand_dims(pred, 1)

        for _ in range(num_future_steps):
            # Predict next latent parameters
            rnn_output, h_state, c_state = self.predictor.lstm_cell(
                current_input, initial_state=[h_state, c_state], training=False
            )
            pred = self.predictor.output_layer(rnn_output)

            # Split into mean and log_var
            pred_mean = pred[:, :latent_dim]
            pred_log_var = pred[:, latent_dim:]

            # Sample from predicted distribution
            z_sample = self._sample(pred_mean, pred_log_var)

            # Decode to image
            img = self.decoder(z_sample)
            future_predictions.append(img)

            # Use prediction as next input
            current_input = tf.expand_dims(pred, 1)

        # Stack predictions
        return tf.stack(future_predictions, axis=1)


class ScheduledSamplingCallback(Callback):
    """
    Callback that gradually reduces the teacher forcing ratio during training.

    This callback implements a curriculum learning schedule for the scheduled
    sampling predictor. It starts with a high teacher forcing ratio (typically 1.0,
    meaning the model always receives ground truth inputs during training) and
    gradually decays this ratio over the course of training according to a specified
    schedule.

    The decay follows a multiplicative schedule: every `decay_every` epochs, the
    current ratio is multiplied by `decay_rate`, until it reaches `min_ratio`.
    This gradual transition helps the model learn robust predictions by slowly
    exposing it to its own errors rather than experiencing an abrupt shift from
    perfect inputs to imperfect ones.

    Arguments
    initial_ratio       Starting teacher forcing ratio. A value of 1.0 means
                        the model always uses ground truth during training.
                        Default is 1.0.
    decay_rate          Multiplicative decay factor applied to the ratio.
                        For example, 0.95 means the ratio is reduced by 5%
                        each time it decays. Default is 0.95.
    decay_every         Number of epochs between each decay step. Default is 5.
    min_ratio           Minimum ratio to maintain. The ratio will never decay
                        below this value. A value of 0.0 means eventually the
                        model will use only its own predictions. Default is 0.0.
    """

    def __init__(
        self, initial_ratio=1.0, decay_rate=0.95, decay_every=5, min_ratio=0.0
    ):
        super().__init__()
        self.initial_ratio = initial_ratio
        self.decay_rate = decay_rate
        self.decay_every = decay_every
        self.min_ratio = min_ratio

    def on_train_begin(self, logs=None):  # pylint:disable=arguments-differ
        """
        Initialize the teacher forcing ratio at the start of training.

        This method is called automatically by Keras when training begins. It sets
        the predictor's teacher forcing ratio to the initial value specified when
        the callback was created.

        Argument
        logs        Dictionary containing training metrics (unused in this method).
        """
        self.model.predictor.teacher_forcing_ratio.assign(self.initial_ratio)
        print(f"\nStarting with teacher forcing ratio: {self.initial_ratio:.4f}")

    def on_epoch_end(self, epoch, logs=None):  # pylint:disable=arguments-differ
        """
        Decay the teacher forcing ratio at the end of specified epochs.

        This method is called automatically by Keras at the end of each epoch. If
        the current epoch number is a multiple of `decay_every`, it multiplies the
        current teacher forcing ratio by `decay_rate`, ensuring it does not fall
        below `min_ratio`.

        Arguments
        epoch       Zero-indexed epoch number (e.g., epoch 0 is the first epoch).
        logs        Dictionary containing training metrics (unused in this method).
        """
        if (epoch + 1) % self.decay_every == 0:
            current_ratio = self.model.predictor.teacher_forcing_ratio.numpy()
            new_ratio = max(self.min_ratio, current_ratio * self.decay_rate)
            self.model.predictor.teacher_forcing_ratio.assign(new_ratio)
            print(f"\nEpoch {epoch+1}: Teacher forcing ratio = {new_ratio:.4f}")
