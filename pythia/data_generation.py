"""
This module handles data generation, i.e. producing the training data that
the models are then trained on.

The underlying physics simulation engine is `pymunk` and
the visual representation of the state of the physical simulation
is governed by `matplotlib`.
"""

import time

from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import pymunk

from matplotlib import patches
from PIL import Image


def generate(  # pylint:disable=too-many-arguments, too-many-locals
    n_samples: int,
    ensure_no_collisions: bool = False,
    monitor_progress: bool = False,
    monitoring_batch_size: int = 1_000,
    physical_width: float = 256.0,
    physical_height: float = 256.0,
    wall_thickness: float = 0.0,
    ball_radius: float = 20.0,
    time_delta_snapshots: float = 0.1,
    time_delta_simulation: float = 0.001,
    max_timestep: float = 2.1,
    min_speed: float = 100.0,
    max_speed: float = 200.0,
    image_width: float = 1024,
    image_height: float = 1024,
    compressed_image_width: float = 32,
    compressed_image_height: float = 32,
) -> tuple:
    """
    Generate the prescribed number of samples. The training samples are stored
    in `images` and each sample is a sequence of images capturing the state of
    the physical simulation at evenly spaced time intervals.

    The simulation consists of a single round ball placed uniformly at random in
    a closed box. The ball is given a random initial velocity and left to bounce
    around that box perfectly elastically.

    Additional data is also stored in the other arrays returned by the function,
    such as `positions` and `velocities`, which may be used for example to select
    samples according to charactersitics of the initial velocity
    or position trajectories (it may be useful to select samples corresponding to
    simulations where there is no collisions between the ball and the walls).

    Arguments
    n_samples                   The number of independent samples to generate.
    ensure_no_collisions        Boolean. Determines whether or not the trajectories
                                generate should be forced to avoid collisions between
                                the ball and the walls.
    monitor_progress            Generating samples can be a time-consuming process.
                                This argument is used to control whether or not
                                a progress message is printed out informing the user
                                of how many samples have been generated so far and
                                how long that took.
    monitoring_batch_size       This argument controls how often the progress message
                                is printed. It will be printed
                                every `monitoring_batch_size`-many samples.
    physical_width              The width of the area within which
                                the physical simulation takes place.
    physical_height             The height of the area within which
                                the physical simulation takes place.
    wall_thickness              The thickness of the walls bounding the simulation area.
    ball_radius                 The radius of the ball.
    time_delta_snapshots        The time intervals at which snapshots are recorded.
    time_delta_simulation       The time intervals used internally by the physics engine
                                `pymunk` to step the simulation forward in time.
                                The smaller this is, the more physically-accurate
                                the simulation is, but the longer it will take
                                to generate samples.
    max_timestep                The time horizon on which to run the simulation.
                                Simulations always start at `t = 0`.
    min_speed                   When randomly generating the initial velocity,
                                its magntitude is bounded below by `min_speed`.
                                See `generate_velocity` for more information
                                on how the initial velocity is generated.
    max_speed                   When randomly generating the initial velocity,
                                its magntitude is bounded above by `max_speed`.
                                See `generate_velocity` for more information
                                on how the initial velocity is generated.
    image_width                 The width (in pixels) of the image used to represent
                                visually the state of the simulation.
    image_height                The height (in pixels) of the image used to represent
                                visually the state of the simulation.
    compressed_image_width      The width (in pixels) of the compressed image used
                                to store snapshots of the simulation.
                                Typically this is (much) smaller than `image_width`
                                to ensure that the array `images` is not too large.
    compressed_image_height     The height (in pixels) of the compressed image used
                                to store snapshots of the simulation.
                                Typically this is (much) smaller than `image_height`
                                to ensure that the array `images` is not too large.

    Returns
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
    # Initialize the physical simulation and its visual representation
    space = _initialize_simulation_space(
        physical_width, physical_height, wall_thickness
    )
    figure, axes = _initialize_visual_representation(
        physical_width, physical_height, image_width, image_height
    )

    # Initialize the ball
    default_position = (physical_width / 2, physical_height / 2)
    ball_body, ball_visual_representation, space, axes = _initialize_ball(
        space, axes, default_position, ball_radius
    )

    # Initialize the output arrays
    n_snapshot_timesteps = int(np.ceil(max_timestep / time_delta_snapshots))
    snapshot_timesteps = time_delta_snapshots * np.arange(n_snapshot_timesteps)
    velocities = np.zeros(shape=(n_samples, 2))
    positions = np.zeros(shape=(n_samples, n_snapshot_timesteps, 2))
    images = np.zeros(
        shape=(
            n_samples,
            n_snapshot_timesteps,
            compressed_image_width,
            compressed_image_height,
            1,
        ),
        dtype=np.uint8,
    )

    # Generate the samples
    start_time = time.time()
    for sample_index in range(n_samples):

        # Monitor progress
        if monitor_progress and (sample_index % monitoring_batch_size == 0):
            elapsed_time_in_minutes = (time.time() - start_time) / 60
            print(
                f"Progress: {sample_index}/{n_samples}, "
                f"elapsed time: {elapsed_time_in_minutes:.1f} minutes."
            )

        # Initialize the ball
        min_distance = ball_radius + wall_thickness
        position = _generate_position(physical_width, physical_height, min_distance)
        ball_body.position = position

        # If we seek to only generate trajectories that avoid collisions,
        # we keep the position generated above but randomly generated independent
        # velocities until a valid velocity is obtained.
        while True:
            velocity = _generate_velocity(min_speed, max_speed)
            if not ensure_no_collisions or _trajectory_contains_no_collisions(
                position,
                velocity,
                snapshot_timesteps[-1],
                physical_width,
                physical_height,
                min_distance,
            ):
                break
        ball_body.velocity = velocity

        # Record the initial velocity
        velocities[sample_index] = np.array(velocity)

        # Run the simulation
        snapshot_index = 0
        for timestep in np.arange(0, max_timestep, time_delta_simulation):
            space.step(time_delta_simulation)
            # Capture snapshots when appropriate
            if (
                snapshot_index < n_snapshot_timesteps
                and timestep >= snapshot_timesteps[snapshot_index]
            ):
                # Record the position
                positions[sample_index, snapshot_index] = np.array(ball_body.position)

                # Reset the visual representation
                for patch in axes.patches:
                    patch.remove()
                # Make the visual representation
                ball_visual_representation.center = np.array(ball_body.position)
                axes.add_patch(ball_visual_representation)

                # Save the rendering in `images`
                images[sample_index, snapshot_index, :, :, 0] = (
                    _save_visual_representation_to_array(
                        figure, compressed_image_width, compressed_image_height
                    )
                )

                # Start waiting for the next snapshot
                snapshot_index += 1

    plt.close()

    return snapshot_timesteps, positions, velocities, images


def _initialize_simulation_space(
    width: float, height: float, wall_thickness: float
) -> pymunk.space.Space:
    """
    Initialize the physical simulation space by creating it and adding the walls to it.

    Note that here `width` and `height` refer to the dimensions of the area
    in which the physical simulation will take place.

    Arguments
    width           The width of the area enclosed by the walls and within which
                    the simulation will take place.
    height          The height of the area enclosed by the walls and within which
                    the simulation will take place.
    wall_thickness  The thickness of the walls bounding the simulation area.

    Returns
    space           A `pymunk.space.Space` object which represents the space in which
                    the physical simulation takes place.
    """
    space = pymunk.Space()

    # Create the walls
    for start, end in (
        ((0, 0), (width, 0)),
        ((0, 0), (0, height)),
        ((width, 0), (width, height)),
        ((0, height), (width, height)),
    ):
        wall = pymunk.Segment(space.static_body, start, end, radius=wall_thickness)
        wall.elasticity = 1
        space.add(wall)

    return space


def _initialize_visual_representation(
    physical_width: float, physical_height: float, image_width: int, image_height: int
) -> tuple:
    """
    Initialize a visual representation of the physical simulation.

    Arguments
    physical_width      The width of the area within which the physical simulation
                        takes place.
    physical_height     The height of the area within which the physical simulation
                        takes place.
    image_width         The width of the image used to represent visually
                        the physical simulation.
    image_height        The height of the image used to represent visually
                        the physical simulation.

    Returns
    figure              `matplotlib.figure.Figure` object containing
                        the overall visual representation.
    axes                `matplotlib.axes._axes.Axes` object containing, essentially,
                        the details of the visual representation.


    Note that here `width` and `height` refer to the dimensions of the area
    in which the physical simulation will take place.

    Note that here `width` and `height` refer to the size (in pixels) of the image
    used to represent the underlying physical simulation.
    """
    figure, axes = plt.subplots(figsize=(image_width / 100, image_height / 100))
    axes.set(
        xlim=(0, physical_width),
        ylim=(0, physical_height),
        aspect="equal",
    )
    axes.axis("off")
    return figure, axes


def _initialize_ball(  # pylint:disable=too-many-arguments
    space: pymunk.space.Space,
    axes: "matplotlib.axes._axes.Axes",
    position: tuple[float, float],
    radius: float,
    mass: float = 1.0,
    moment: float = 10.0,
    elasticity: float = 1.0,
) -> tuple:
    """
    Given an object `space` representing the physical simulation and
    an object `axes` representing its visual representation,
    add a ball with the prescribed characteristics to that simulation and
    to its visual representation.

    Arguments
    space               A `pymunk.space.Space` object which represents the space
                        in which the physical simulation takes place.
    axes                `matplotlib.axes._axes.Axes` object containing, essentially,
                        the details of the visual representation.
    position            The position of the ball.
    radius              The radius of the ball.
    mass                The mass of the ball.
    moment              The moment of inertia of the ball (which is a scalar since
                        we are in two dimensions).
    elasticity          The elasticity coefficient of the ball.
                        `elasticity = 1` corresponds to a perfectly elastic ball,
                        which means that the ball will not lose any kinetic energy
                        when colliding with another object (such as a wall).
                        `elasticity = 0` corresponds to a perfectly inelastic ball,
                        which means that the ball will lose all of its kinetic energy
                        when colliding with another object (such as a wall), and
                        so will typically stop moving.

    Returns
    body                A `pymunk.body.Body` object which represents the physical body
                        of the ball and its characteristics properties
                        as it lives within the physical simulation described by `space`.
    visual_representation   A `matplotlib.patches.Circle` object which describes
                        the visual representation of the ball.
    space               A `pymunk.space.Space` object which represents
                        the space in which the physical simulation takes place.
    axes                `matplotlib.axes._axes.Axes` object containing, essentially,
                        the details of the visual representation.
    """
    # Create the ball
    body = pymunk.Body(mass=mass, moment=moment)
    shape = pymunk.Circle(body, radius=radius)
    shape.elasticity = elasticity
    # Add the ball to the physical simulation
    space.add(body, shape)
    # Add the ball to the visual representation
    # (place it in the center by default)
    visual_representation = patches.Circle(xy=position, radius=radius, color="black")
    axes.add_patch(visual_representation)
    return body, visual_representation, space, axes


def _generate_position(
    width: float, height: float, min_distance: float
) -> tuple[float, float]:
    """
    Returns a position chosen uniformly at random in the rectangle
        `[min_distance, width - min_distance] x [min_distance, height - min_distance]`.

    The argument `min_distance` is used to ensure that the object that will then
    be placed at the generated position will not clip the walls by being too close
    to them.

    Note that here `width` and `height` refer to the dimensions of the area
    in which the physical simulation will take place.

    Arguments
    width           Width of the rectangle in which the position is to be found,
                    while staying at least `min_distance` away from its edges.
    height          Height of the rectangle in which the position is to be found,
                    while staying at least `min_distance` away from its edges.
    min_distance    Minimal distance between the position generated and the boundary
                    of the rectangle `[0, width] x [0, height]`.

    Returns
    position        A pair of floats describing the `x` and `y` position of an object.
    """
    position = (
        uniform(min_distance, width - min_distance),
        uniform(min_distance, height - min_distance),
    )
    return position


def _generate_velocity(min_speed: float, max_speed: float) -> tuple[float, float]:
    """
    Returns a velocity constructed by choosing its magnitude uniformly at random
    in the interval `[min_speed, max_speed]` and by choosing its angle uniformly
    at random in the interval `[0, 2*pi]`.

    Arguments
    min_speed   The minimum magnitude of the velocity vector generated.
    max_speed   The maximum magnitude of the velocity vector generated.

    Returns
    velocity    A pair of floats describing the velocity of an objet.
    """
    speed = uniform(min_speed, max_speed)
    angle = uniform(0, 2 * np.pi)
    velocity = (speed * np.cos(angle), speed * np.sin(angle))
    return velocity


def _save_visual_representation_to_array(
    figure: "matplotlib.figure.Figure",
    width: int,
    height: int,
) -> np.ndarray:
    """
    Save the current state of the visual representation captured by `figure`
    as a `numpy` array representation of a grayscale image.

    We save the image as an array of unsigned 8-bit integers. In other words,
    the entries of `images` are integers between 0 and 255 (0 and 255 included).
    This data type is chosen to minimize the space that the `images` array
    takes on disk when saved or loaded.

    Arguments
    figure      `matplotlib.figure.Figure` object containing
                the overall visual representation.
    width       The width of the image saved. Typically this is small, and
                thus requires the visual representation described by figure
                to be compressed.
    height      The height of the image saved. Typically this is small, and
                thus requires the visual representation described by figure
                to be compressed.

    Returns
    image       `numpy` array of shape `(width, height)` capturing `figure` as an image.
    """
    figure.canvas.draw()
    # Get raw pixel data from `figure` as a 1D `numpy` array.
    image = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8)
    # Reshape the array to be of shape `(width, height, 4)`,
    # which is standard for an image of size `width`x`height,
    # with four channels (three RGB and one alpha/transparency).
    image = image.reshape(figure.canvas.get_width_height()[::-1] + (4,))
    # Drop the alpha/transparency channel.
    image = image[:, :, :3]
    # Convert from RGB to grayscale.
    image = np.array(Image.fromarray(image).convert("L"))
    # Resize the grayscale image
    image = np.array(Image.fromarray(image).resize((width, height)))
    return image


def _trajectory_contains_no_collisions(  # pylint: disable=too-many-arguments
    position: tuple[float, float],
    velocity: tuple[float, float],
    time_horizon: float,
    width: float,
    height: float,
    min_distance: float,
) -> bool:
    """
    Determines whether or not a trajectory starting at position `position`
    with initial velocity `velocity` at time `t = 0` will be within, or past,
    a distance `min_distance` of the boundary of the rectangle
        `[0, width] x [0, height]`
    by the time `t = time_horizon`.

    Arguments
    position        Initial position of a point particle.
    velocity        Initial velocity of a point particle.
    time_horizon    The time horizon on which to check whether or no collisions
                    occur. I.e. the time interval considered is
                    `[0, time_horizon]`.
    width           The width of the area within which the point particle moves.
    height          The height of the area within which the point particle moves.
    min_distance    The minimum distance between the trajectory and
                    the boundaries.

    Returns
    boolean equal to `True` if no collisions will occur and `False` otherwise.
    """
    return (
        min_distance < position[0] + time_horizon * velocity[0] < width - min_distance
        and min_distance
        < position[1] + time_horizon * velocity[1]
        < height - min_distance
    )
