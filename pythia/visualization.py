"""
This module contains routines used to visualize the input data, the predictions made,
and compare visually the training data to the predictions made.
"""

import matplotlib.pyplot as plt
import numpy as np


def inspect(
    snapshot_timesteps: list[float],
    positions: np.ndarray,
    velocity: np.ndarray,
    images: np.ndarray,
) -> None:
    """
    Given a training data sample, display a row of plots of the image sequence, with
    the first image on the left and the last image on the right. The exact velocity and
    sequence of exact positions are also printed out (even though the model will not
    have access to that information when training; it is provided for when fine-grained
    control over the samples to include is helpful).

    Arguments
    snapshot_timesteps      List of the times at which the image snapshots are recorded.
                            Provided so that these times can be used as legend for
                            the plots.

                            We will use `n_snapshot_timesteps = len(snapshot_timesteps)`
                            below.
    positions               `numpy` array of shape `(n_snapshot_timesteps, 2)` where
                            `positions[i]` is the position, in the two-dimensional
                            plane, of the ball at time `snapshot_timesteps[i]`.
    velocity                `numpy` array of shape `(2,)` corresponding to the initial
                            velocity of the ball.
    images                  `numpy` array of shape
                            `(n_snapshot_timesteps, width, height, 1)`
                            where `images[i]` is the image recorded at time
                            `snapshot_timesteps[i]`.

                            The last dimension of the `images` array corresponds to
                            the fact that these images are grayscale, and so is encoded
                            in a single grayscale channel.

    """

    n_snapshot_timesteps = len(snapshot_timesteps)
    _, width, height, _ = images.shape

    # Plots
    _, axes = plt.subplots(
        1, n_snapshot_timesteps, figsize=(3 * n_snapshot_timesteps, 3)
    )
    for snapshot_index, (axis, snapshot_time) in enumerate(
        zip(axes, snapshot_timesteps)
    ):
        axes[snapshot_index].imshow(images[snapshot_index], cmap="gray")
        axis.set(xlim=(0, width), ylim=(height, 0), title=f"t = {snapshot_time:.1f}")
        axis.axis("off")
    plt.show()

    # Initial velocity
    print(f"Initial velocity:        {velocity}")

    # Positions
    for snapshot_index, snapshot_time in enumerate(snapshot_timesteps):
        print(f"Position at t = {snapshot_time:.1f}: {positions[snapshot_index]}")
