"""This module contains routines used in the demo notebook."""

from importlib import resources

import numpy as np


def load_demo_data() -> tuple:
    """
    Load the demonstration data, returning a tuple
    `(snapshot_timesteps, positions, velocities, images)`.

    Below we will use `n_samples` to denote the number of samples loaded in this manner.

    Returns
    snapshot_timesteps      List of the times at which the image snapshots are recorded.
                            Provided so that these times can be used as legend for
                            the plots.
    positions               `numpy` array of shape
                            `(n_samples, n_snapshot_timesteps, 2)`
                            where `positions[i, j]` is the position, in the two-dimensional
                            plane, of the ball at time `snapshot_timesteps[j]`
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
        """Load the `numpy` array located at `pythia/demo/filename`."""
        path = resource.files("pythia").joinpath("demo/" + filename)
        return np.load(path)

    snapshot_times = np.load("demo_snapshot_timesteps.npy")
    positions = np.load("demo_positions.npy")
    velocities = np.load("demo_velocities.npy")
    images = np.load("demo_images.npy")

    return snapshot_timesteps, positions, velocities, images
