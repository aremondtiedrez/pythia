"""
This module defines the deep learning models used in PyThia,
which are built using `keras`.
"""

import numpy as np

from keras import Model, ops
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    Reshape,
)
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

        # Build the three components of the model
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.predictor = self._build_predictor()

        # `shape_before_flattening` is initialized when building the encoder and
        # ensures that the output of the encoder is consistent with the input
        # of the decoder
        self._shape_before_flattening = None

    def _build_encoder(self) -> Model:
        """
        Build the encoder.

        The encoder is a convolutional neural network which maps images
        to latent space embeddings.

        Returns
        `tensorflow.keras.Model` object describing the encoder.
        """
        input_img = Input(shape=self.img_shape)

        x = Conv2D(32, 3, padding="same", activation="relu")(input_img)
        x = Conv2D(64, 3, padding="same", activation="relu", strides=2)(x)
        x = Conv2D(64, 3, padding="same", activation="relu")(x)
        x = Conv2D(64, 3, padding="same", activation="relu")(x)

        self._shape_before_flattening = x.shape[1:]

        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)

        # These are the mean and log-variance of the embedding
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

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
        predictor_input = Input(shape=(4 * self.latent_dim,))
        predictor_output = Dense(2 * self.latent_dim, activation="linear")(
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
        decoder_input = Input(shape=(self.latent_dim,))

        x = Dense(np.prod(self._shape_before_flattening), activation="relu")(
            decoder_input
        )
        x = Reshape(self._shape_before_flattening)(x)
        x = Conv2DTranspose(32, 3, padding="same", activation="relu", strides=2)(x)
        x = Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)

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
