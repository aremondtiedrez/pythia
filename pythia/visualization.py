"""
This module contains routines used to visualize the input data, the predictions made,
and compare visually the training data to the predictions made.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation

from . import models


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
            displayed_images[snapshot_index] = _add_walls(
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
    plt.close()

    # Initial velocity
    print(f"Initial velocity:    {np.array2string(velocity, precision=2)}")

    # Positions
    for snapshot_index, snapshot_time in enumerate(snapshot_timesteps):
        print(
            f"Position at t = {snapshot_time:.1f}: "
            f"{np.array2string(positions[snapshot_index], precision=2)}"
        )


def inspect_sample(
    index: int,
    snapshot_timesteps: list[float],
    positions: np.ndarray,
    velocities: np.ndarray,
    images: np.ndarray,
    **kwargs,
) -> None:
    """
    Thin wrapper around `inspect` used when the data stored in `positions`,
    `velocities`, and `images` consists of several samples (by contrast, the method
    `inspect` handles a single sample at a time).

    Arguments
    index                       index of the sample to observe, which is an index
                                along the first dimension of `positions`, `velocities`,
                                and `images`.
    snapshot_timesteps          List of the times at which the image snapshots
                                are recorded. Provided so that these times
                                can be used as legend for the plots.
    positions                   `numpy` array of shape
                                `(n_samples, n_snapshot_timesteps, 2)`
                                where `positions[i, j]` is the position,
                                in the two-dimensional plane, of the ball
                                at time `snapshot_timesteps[j]`
                                corresponding to the `i`-th sample.
    velocities                  `numpy` array of shape `(n_samples, 2)`
                                where `velocities[i]` corresponds to
                                the initial velocity of the ball in the `i`-th sample.
    images                      `numpy` array of shape
                                `(n_samples, n_snapshot_timesteps, width, height, 1)`
                                where `images[i, j]` is the image recorded at time
                                `snapshot_timesteps[j]` for the `i`-th sample.

                                The last dimension of the `images` array corresponds to
                                the fact that these images are grayscale, and
                                so is encoded in a single grayscale channel.

                                The `images` array has `images.dtype == np.uint8`,
                                i.e. its entries are unsigned 8-bit integers.
                                In other words, the entries of `images` are integers
                                between 0 and 255 (0 and 255 included).
                                This data type is chosen to minimize the space
                                that the `images` array takes on disk when saved or
                                loaded.
    """
    inspect(
        snapshot_timesteps, positions[index], velocities[index], images[index], **kwargs
    )


def memoryless_prediction(  # pylint: disable=too-many-arguments
    snapshot_timesteps: list[float],
    img0: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    memoryless_model: "pythia.models.Memoryless",
    display_walls: bool = False,
) -> None:
    """
    Given a training sample for the memoryless model and a trained instance
    of the memoryless model, inspect visually the performance of that model
    on prediction.

    Arguments
    snapshot_timesteps      List of the times at which the image snapshots are recorded.
                            Provided so that these times can be used as legend for
                            the plots.
    img0                    Image from the immediate past, captured at, say, `t = 0`.
                            Provided as a `numpy` array of shape
                            `(width, height, 1)`, where the last dimension being `1`
                            indicates that this image is grayscale.
    img1                    Image from the immediate past, captured at, say, `t = 1`.
                            Provided as a `numpy` array of shape
                            `(width, height, 1)`, where the last dimension being `1`
                            indicates that this image is grayscale.
    img2                    Image in the immediate future, captured at, say, `t = 2`.
                            Provided as a `numpy` array of shape
                            `(width, height, 1)`, where the last dimension being `1`
                            indicates that this image is grayscale.
    memoryless_model        Model defined in `models.Memoryless` which takes as input
                            two images in the immediate past, such as `img0` and `img1`,
                            and predicts the image in the immediate future.
    display_walls           By default, the raw image is displayed, which does
                            not include the walls delimiting the space in which
                            the physical simulation takes place. This boolean argument
                            can be used to display the walls as part of
                            the visualization.
    """

    width, height, _ = img0.shape

    predicted_image = memoryless_model.predict_next_image(
        np.expand_dims(img0, axis=0),
        np.expand_dims(img1, axis=0),
    )[0]

    images_to_plot = (
        img0,
        img1,
        predicted_image,
        img2,
        img2 - predicted_image,
    )
    if display_walls:
        images_to_plot = (_add_walls(image) for image in images_to_plot)
    plot_titles = (
        f"Past image at t = {snapshot_timesteps[0]}",
        f"Past image at t = {snapshot_timesteps[1]}",
        f"Prediction future image at t = {snapshot_timesteps[2]}",
        f"True image at t = {snapshot_timesteps[2]}",
        "Difference between\nthe true and predicted futures",
    )

    _, axes = plt.subplots(1, 5, figsize=(21, 3))
    for axis, image, title in zip(axes, images_to_plot, plot_titles):
        axis.imshow(image, cmap="gray")
        axis.set(xlim=(0, width), ylim=(height, 0), title=title)
        axis.axis("off")
    plt.show()
    plt.close()


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
        frames = [_add_walls(frame) for frame in frames]

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


def reconstruction(
    true_image: np.ndarray,
    encoder: "keras.Model",
    decoder: "keras.Model",
    display_walls: bool = False,
) -> None:
    """
    Given a true image and an encoder-decoder pair,
    use that pair to reconstruct the image and then visualize the difference.

    Arguments
    true_image      `numpy` array of shape `(width, height, 1)`, since the image
                    is grayscale.
    encoder         Model of class `keras.Model` used to encode the image
                    into a latent space embedding
    decoder         Model of class `keras.Model` used to decode a latent space
                    embedding into an image.
    display_walls   By default, the raw image is displayed, which does not include
                    the walls delimiting the space in which the physical simulation
                    takes place. This boolean argument can be used to display
                    the walls as part of the visualization.
    """
    width, height, _ = true_image.shape

    # The embedding contains both the mean and the logarithmic variance,
    # so we only feed the first half of the embedding, i.e. the mean, to the decoder.
    embedding, _ = encoder(np.expand_dims(true_image, axis=0), verbose=0)
    reconstructed_image = decoder(embedding, verbose=0)[0]

    images_to_plot = (true_image, reconstructed_image, true_image - reconstructed_image)
    if display_walls:
        images_to_plot = (_add_walls(image) for image in images_to_plot)
    plot_titles = ("Truth", "Prediction", "Difference")
    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    for axis, image, title in zip(axes, images_to_plot, plot_titles):
        axis.imshow(image, cmap="gray")
        axis.set(xlim=(0, width), ylim=(height, 0), title=title)
        axis.axis("off")
    plt.show()
    plt.close()


def prediction(  # pylint: disable=missing-function-docstring, too-many-arguments, too-many-locals
    encoder,
    predictor,
    decoder,
    latent_dim,
    snapshot_timesteps,
    ground_truth_images,
    n_past_steps,
    n_future_steps,
):

    n_timesteps = ground_truth_images.shape[0]
    if n_timesteps != n_past_steps + n_future_steps:
        raise ValueError(
            "n_past_steps and n_future_steps must add up to "
            "the number of timesteps in ground_truth_images"
        )

    # Make predictions
    images_in_past = ground_truth_images[:n_past_steps]
    predicted_images = np.array(
        models.predict_future(
            encoder,
            predictor,
            decoder,
            latent_dim,
            np.expand_dims(images_in_past, axis=0),
            n_future_steps,
        )
    )[0]

    # Create plots
    figure, axes = plt.subplots(3, n_timesteps, figsize=(2 * n_timesteps, 5))
    for timestep_index in range(n_timesteps):
        axes[0, timestep_index].imshow(ground_truth_images[timestep_index], cmap="gray")
        if timestep_index >= n_past_steps:
            axes[1, timestep_index].imshow(
                predicted_images[timestep_index - n_past_steps], cmap="gray"
            )
            axes[2, timestep_index].imshow(
                ground_truth_images[timestep_index]
                - predicted_images[timestep_index - n_past_steps],
                cmap="gray",
            )
    # Row labels
    row_labels = ("Truth", "Prediction", "Difference")
    for pos_index, row_label in enumerate(row_labels):
        # Get the bounding box of the leftmost subplot in this row
        bbox = axes[pos_index, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2  # Vertical center of the row
        figure.text(0.02, y_center, row_label, va="center")
    # Column labels
    col_labels = (f"t = {timestep:.1f}" for timestep in snapshot_timesteps)
    for axis, col_label in zip(axes[0, :], col_labels):
        axis.set_title(col_label)
    # Hide axes
    for axis in axes.flatten():
        axis.axis("off")

    plt.show()
    plt.close()


def _add_walls(input_image: np.ndarray, wall_location=4) -> np.ndarray:
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
    output_image = np.array(input_image).copy()
    # Top wall
    output_image[wall_location, wall_location:-wall_location] = 0
    # Bottom wall
    output_image[-wall_location, wall_location:-wall_location] = 0
    # Left wall
    output_image[wall_location:-wall_location, wall_location] = 0
    # Right wall
    output_image[wall_location : -wall_location + 1, -wall_location] = 0
    return output_image
