"""
This module contains routines used to visualize the input data, the predictions made,
and compare visually the training data to the predictions made.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation


def inspect(
    snapshot_timesteps: list[float],
    positions: np.ndarray,
    velocity: np.ndarray,
    images: np.ndarray,
    display_walls: bool = False,
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
    display_walls           By default, the raw image is displayed, which does
                            not include the walls delimiting the space in which
                            the physical simulation takes place. This boolean argument
                            can be used to display the walls as part of
                            the visualization.
    """

    n_snapshot_timesteps = len(snapshot_timesteps)
    _, width, height, _ = images.shape

    # If desired, add walls to the images
    displayed_images = images.copy()
    if display_walls:
        for snapshot_index in range(n_snapshot_timesteps):
            displayed_images[snapshot_index] = add_walls(
                displayed_images[snapshot_index]
            )

    # Plots
    _, axes = plt.subplots(
        1, n_snapshot_timesteps, figsize=(3 * n_snapshot_timesteps, 3)
    )
    for snapshot_index, (axis, snapshot_time) in enumerate(
        zip(axes, snapshot_timesteps)
    ):
        axes[snapshot_index].imshow(displayed_images[snapshot_index], cmap="gray")
        axis.set(xlim=(0, width), ylim=(height, 0), title=f"t = {snapshot_time:.1f}")
        axis.axis("off")
    plt.show()

    # Initial velocity
    print(f"Initial velocity:        {velocity}")

    # Positions
    for snapshot_index, snapshot_time in enumerate(snapshot_timesteps):
        print(f"Position at t = {snapshot_time:.1f}: {positions[snapshot_index]}")


def create_animation(images: np.ndarray, display_walls: bool = False) -> FuncAnimation:
    """
    Given a training data sample, create a `matplotlib.animation.FuncAnimation` object
    which is able to be rendered within a notebook or saved as a GIF.

    Arguments
    images                  `numpy` array of shape
                            `(n_snapshot_timesteps, width, height, 1)`
                            where `images[i]` is the image recorded at time
                            `snapshot_timesteps[i]`.

                            The last dimension of the `images` array corresponds to
                            the fact that these images are grayscale, and so is encoded
                            in a single grayscale channel.
    display_walls           By default, the raw image is displayed, which does
                            not include the walls delimiting the space in which
                            the physical simulation takes place. This boolean argument
                            can be used to display the walls as part of
                            the visualization.

    Returns
    animation               A `matplotlib.animation.FuncAnimation` object
                            which can be used to render the corresponding
                            animation/video within a notebook or save it as a GIF file.
    """
    n_snapshot_timesteps, width, height, _ = images.shape

    frames = [images[snapshot_index] for snapshot_index in range(n_snapshot_timesteps)]

    if display_walls:
        frames = [add_walls(frame) for frame in frames]

    figure, axes = plt.subplots(figsize=(4, 4))
    frame = axes.imshow(frames[0], cmap="gray")
    axes.set(xlim=(0, width), ylim=(height, 0))
    axes.axis("off")

    def update(frame_index):
        frame.set_array(frames[frame_index])
        return [frame]

    animation = FuncAnimation(
        figure,
        update,
        frames=n_snapshot_timesteps,
        interval=500,
        blit=True,
        repeat=True,
    )
    plt.close()
    return animation


def add_walls(input_image: np.ndarray, wall_location=4) -> np.ndarray:
    """
    This function adds walls back to the visualization of a training image.

    Background: for simplicity, the model is fed images *without* walls when training.
    This avoids dedicating resources to embedding and reconstructing the walls, allowing
    the model to focus on finding an embedding which can be used to predict the dynamics
    of the ball as it bounces off the walls. Nonetheless, when the images are provided
    to a user for visualization, it is helpful to add the walls back into
    the visualization. This avoids the ball appearing to bounce against an invisible
    object.

    The argument `wall_location` is chosen by hand to be equal to 4 as that
    is the correct value to use for the default image size used throughout the project
    (which are 32x32 images, compressed down from the original 1024x1024 images).
    It corresponds to the precise pixel, in the image, where the wall ought to be.
    A different value of `wall_location` would have to be used for images
    of a different size

    Argument
    input_image     A `numpy` array of size `(width, height, 1)` representing
                    a grayscale image.

                    The last dimension of the `images` array corresponds to
                    the fact that these images are grayscale, and so is encoded
                    in a single grayscale channel.
    """
    output_image = input_image.copy()
    # Top wall
    output_image[wall_location, wall_location:-wall_location] = 0
    # Bottom wall
    output_image[-wall_location, wall_location:-wall_location] = 0
    # Left wall
    output_image[wall_location:-wall_location, wall_location] = 0
    # Right wall
    output_image[wall_location : -wall_location + 1, -wall_location] = 0
    return output_image
