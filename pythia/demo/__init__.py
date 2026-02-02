"""This module contains routines used in the demo notebook."""

from importlib import resources

import numpy as np

from .. import models


def load_data(kind: str = "demo") -> tuple:
    """
    Load the demonstration data, returning a tuple
    `(snapshot_timesteps, positions, velocities, images)`.

    Below we will use `n_samples` to denote the number of samples loaded in this manner.

    Argument
    kind                    Which data to load.
                            `demo` (default)    A small number of samples, issued from
                                                the test data, for quick loading.
                            `collisionless`     Collisionless training samples.
                            `training`          Training samples.
                            `test`              Testing samples.

    Returns
    snapshot_timesteps      List of the times at which the image snapshots are recorded.
                            Provided so that these times can be used as legend for
                            the plots.
    positions               `numpy` array of shape
                            `(n_samples, n_snapshot_timesteps, 2)`
                            where `positions[i, j]` is the position,
                            in the two-dimensional plane, of the ball
                            at time `snapshot_timesteps[j]`
                            corresponding to the `i`-th sample.
    velocities              `numpy` array of shape `(n_samples, 2)`
                            where `velocities[i]` corresponds to the initial velocity
                            of the ball in the `i`-th sample.
    images                  `numpy` array of shape
                            `(n_samples, n_snapshot_timesteps, width, height, 1)`
                            where `images[i, j]` is the image recorded at time
                            `snapshot_timesteps[j]` for the `i`-th sample.

                            The last dimension of the `images` array corresponds to
                            the fact that these images are grayscale, and so is encoded
                            in a single grayscale channel.

                            The `images` array has `images.dtype == np.uint8`, i.e. its
                            entries are unsigned 8-bit integers. In other words,
                            the entries of `images` are integers between 0 and 255
                            (0 and 255 included). This data type is chosen to minimize
                            the space that the `images` array takes on disk
                            when saved or loaded.
    """

    def load(filename):
        """Load the `numpy` array located at `pythia/demo/data/filename`."""
        path = resources.files("pythia").joinpath("demo/data/" + filename)
        return np.load(path)

    snapshot_timesteps = load(kind + "_snapshot_timesteps.npy")
    positions = load(kind + "_positions.npy")
    velocities = load(kind + "_velocities.npy")
    images = load(kind + "_images.npy")

    return snapshot_timesteps, positions, velocities, images


def load_model(name: str = None) -> "keras.Model":
    """
    Load one of the demonstration models.

    The models that may be loaded are the following.
    name
    memoryless_full         Model mapping two images in the immediate past to
                            the next image in the immediate future. This model
                            is primarily used because its training is used
                            to pre-train the encoder-decoder pair.
    memoryless_encoder      The encoder model which is part of the memoryless
                            model above.
    memoryless_decoder      The decoder model which is part of the memoryless
                            model above.

    Argument
    name    String describing the model to load.

    Returns
    model   `keras.Model` object describing the model loaded.
    """

    path = resources.files("pythia").joinpath("demo/models/" + name + ".weights.h5")

    if name == "memoryless_full":
        model = models.Memoryless()
        model.compile()
        model.load_weights(path)
    elif name == "memoryless_encoder":
        model = models.Memoryless()
        model = model.encoder
        model.compile()
        model.load_weights(path)
    elif name == "memoryless_decoder":
        model = models.Memoryless()
        model = model.decoder
        model.compile()
        model.load_weights()

    return model
