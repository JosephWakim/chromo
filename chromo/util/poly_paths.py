"""Functions for specification of initial polymer paths.
"""

from typing import Callable, Tuple, Optional, List

import numpy as np
from numba import njit


def coordinates_in_x_y(
    num_beads: int,
    bead_length: float,
    shape_func: Callable[[float], float],
    step_size: float
) -> np.ndarray:
    """Find bead coordiantes about path in x, y plane w/ fixed contour length.

    Notes
    -----
    Get coordinates in the x, y plane which splits the path of the polymer into
    segments of equal path length.

    TODO: If we want to have variable linker lengths, the spacing of monomeric
    units must be different, and this must be accounted for when selecting
    x-positions of beads.

    Parameters
    ----------
    num_beads : int
        Number of monomer units on the polymer
    bead_length : float or (N,) array_like of float
        The amount of polymer path length between this bead and the next
        bead. For now, a constant value is assumed (the first value if an
        array is passed).
    shape_func : Callable[[float], float]
        Shape of the polymer where z = 0 and y = f(x)
    step_size : float
        Step size for numerical evaluation of contour length when
        domain of the shape function used to initialize polymer

    Returns
    -------
    np.ndarray
        Coordinates of each point with path defined by `shape_func`
    """
    r = np.zeros((num_beads, 3))
    for i in range(1, num_beads):
        x = r[i-1, 0]
        arc_length = 0

        while arc_length < bead_length:
            previous_x = x
            previous_arc_length = arc_length

            dy_dx = numerical_derivative(shape_func, previous_x, step_size)
            x += step_size
            arc_length += np.sqrt(1 + (dy_dx)**2) * step_size

        r[i, 0] = np.interp(
            bead_length, [previous_arc_length, arc_length], [previous_x, x]
        )
        r[i, 1] = shape_func(r[i, 0])
    return r


def coordinates_in_x_y_z(
    num_beads: int,
    bead_length: float,
    shape_func_x: Callable[[float], float],
    shape_func_y: Callable[[float], float],
    shape_func_z: Callable[[float], float],
    step_size: float
) -> Tuple[np.ndarray, List[float]]:
    """Generate coordinates for 3D initialization of a polymer.

    Parameters
    ----------
    num_beads : int
        Number of monomer units on the polymer
    bead_length : float or (N,) array_like of float
        The amount of polymer path length between this bead and the next
        bead. For now, a constant value is assumed (the first value if an
        array is passed).
    shape_func_x : Callable[[float], float]
        Parametric functions to obtain the x coordinates of the path
    shape_func_y : Callable[[float], float]
        Parametric functions to obtain the y coordinates of the path
    shape_func_z : Callable[[float], float]
        Parametric functions to obtain the z coordinates of the path
    step_size : float
        Step size for numerical evaluation of contour length when
        domain of the shape function used to initialize polymer

    Returns
    -------
    np.ndarray
        Coordinates of points with path defined by parametric shape functions
    List[float]
        Parameter values corresponding to (x, y, z) points obtained from shape
        functions
    """
    r = np.zeros((num_beads, 3))
    t = 0
    parameter_vals = [t]
    for i in range(num_beads):
        arc_length = 0

        while arc_length < bead_length:
            previous_t = t
            previous_arc_length = arc_length
            dx_dt = numerical_derivative(shape_func_x, previous_t, step_size)
            dy_dt = numerical_derivative(shape_func_y, previous_t, step_size)
            dz_dt = numerical_derivative(shape_func_z, previous_t, step_size)
            t += step_size
            arc_length += np.sqrt(
                dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2
            ) * step_size

        t_true = np.interp(
            bead_length, [previous_arc_length, arc_length], [previous_t, t]
        )
        parameter_vals.append(t_true)
        r[i, 0] = shape_func_x(t_true)
        r[i, 1] = shape_func_y(t_true)
        r[i, 2] = shape_func_z(t_true)

    return r, parameter_vals


def get_tangent_vals_x_y(
    x: np.ndarray,
    shape_func: Callable[[float], float],
    step_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the t3 and t2 vectors at a fixed position along shape_func.

    Parameters
    ----------
    x : np.ndarray
        Vector of independent variable positions at which to evaluate tangent
    shape_func : Callable[[float], float]
        Function defining the shape on which to obtain tangent
    step_size : float
        Step size to use when numerically evaluating tangent vectors

    Returns
    -------
    np.ndarray (N, 3)
        Matrix representing t3 tangents, where rows represent x, y, z
        components of tangent at each point.
    np.ndarray (N, 3)
        Matrix representing t2 tangents, where rows represent x, y, z
        coordinates of tangent vector at each point
    """
    num_beads = len(x)
    t3 = np.zeros((num_beads, 3))
    t2 = np.zeros((num_beads, 3))

    arbitrary_vec_1 = np.array([1, 0, 0])
    arbitrary_vec_2 = np.array([0, 1, 0])

    for i in range(num_beads):
        t3_i = np.array(
            [step_size, numerical_derivative(
                shape_func, x[i], step_size
            ) * step_size]
        )
        t3[i, 0:2] = t3_i / np.linalg.norm(t3_i)
        trial_t2 = np.cross(t3[i, :], arbitrary_vec_1)
        if np.all(trial_t2 == 0):
            trial_t2 = np.cross(t3[i, :], arbitrary_vec_2)
        t2[i, :] = trial_t2 / np.linalg.norm(trial_t2)

    return t3, t2


def get_tangent_vals_x_y_z(
    t: np.ndarray,
    shape_func_x: Callable[[float], float],
    shape_func_y: Callable[[float], float],
    shape_func_z: Callable[[float], float],
    step_size: float,
    r: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the t3 and t2 vectors at a fixed position along shape_func.

    Parameters
    ----------
    t : np.ndarray
        Vector of parameter values at which to evaluate the tangents
    r : Optional[np.ndarray]
        Coordinates of points corresponding to parameter values
    shape_func_x : Callable[[float], float]
        Parametric functions to obtain the x coordinates of the path
    shape_func_y : Callable[[float], float]
        Parametric functions to obtain the y coordinates of the path
    shape_func_z : Callable[[float], float]
        Parametric functions to obtain the z coordinates of the path
    step_size : float
        Step size to use when numerically evaluating tangent vectors

    Returns
    -------
    array_like (N, 3) of double
        Matrix representing t3 tangents, where rows represent x, y, z
        components of tangent at each point.
    array_like (N, 3) of double
        Matrix representing t2 tangents, where rows represent x, y, z
        coordinates of tangent vector at each point
    """
    if r is None:
        r = np.array(
            [
                [shape_func_x(t_i), shape_func_y(t_i), shape_func_z(t_i)]
                for t_i in t
            ]
        )
    num_beads = len(t) - 1
    t3 = np.zeros((num_beads, 3))
    t2 = np.zeros((num_beads, 3))

    arbitrary_vec_1 = np.array([1, 0, 0])
    arbitrary_vec_2 = np.array([0, 1, 0])

    for i in range(0, num_beads):
        if i == 0:
            t3_i = np.array([1, 0, 0])
        else:
            t3_i = np.array(
                [
                    shape_func_x(t[i] + step_size) - r[i, 0],
                    shape_func_y(t[i] + step_size) - r[i, 1],
                    shape_func_z(t[i] + step_size) - r[i, 2]
                ]
            )
        t3[i, :] = t3_i / np.linalg.norm(t3_i)
        trial_t2 = np.cross(t3[i, :], arbitrary_vec_1)
        if np.all(trial_t2 == 0):
            trial_t2 = np.cross(t3[i, :], arbitrary_vec_2)
        t2[i, :] = trial_t2 / np.linalg.norm(trial_t2)

    return t3, t2


def numerical_derivative(
    shape_func: Callable[[float], float],
    point: float,
    step_size: float
) -> float:
    """Numerically evaluate the derivative of `shape_func` at a point.

    Parameters
    ----------
    shape_funct : Callable[[float], float]
        Function from which to evaluate derivative
    point : float
        Point at which to evaluate derivative
    step_size : float
        Step-size to apply when numerically evaluating derivative

    Returns
    -------
    float
        Numerical approximation of derivative at point
    """
    return (shape_func(point + step_size) - shape_func(point)) / step_size


def gaussian_walk(
    num_steps: int,
    step_sizes: np.ndarray
) -> np.ndarray:
    """Generate coordinates for Gaussian random walk w/ fixed path length.

    Parameters
    ----------
    num_steps : int
        Number of steps in the Gaussian random walk
    step_sizes : np.ndarray
        Distances between subsequent points in the random walk

    Returns
    -------
    np.ndarray (N, 3)
        Coordinates of each point in the Gaussian random walk, where rows
        represent individual points and columns give x, y, z coordinates
    """
    steps = np.random.standard_normal((num_steps, 3))
    magnitude_steps = np.linalg.norm(steps, axis=1)
    positions = np.cumsum(
        np.divide(steps, magnitude_steps[:, None]) * step_sizes[:, np.newaxis],
        axis=0
    )
    positions = np.vstack((np.zeros(3), positions))
    return positions


def confined_gaussian_walk(
    num_points: int,
    step_sizes: np.ndarray,
    confine_type: str,
    confine_length: float
) -> np.ndarray:
    """Generate coordinates for Gaussian random walk w/ fixed path length.

    Parameters
    ----------
    num_points : int
        Number of points in the Gaussian random walk
    step_sizes : np.ndarray (N-1,) of float
        Distance between each point in the random walk
    confine_type : str
        Name of the confining boundary. To indicate model w/o confinement,
        enter a blank string for this argument
    confine_length : double
        The lengthscale associated with the confining boundary. Length
        representation specified in function associated w/ `confine_type`

    Returns
    -------
    np.ndarray (N, 3)
        Coordinates of each point in the Gaussian random walk, where rows
        represent individual points and columns give x, y, z coordinates
    """
    points = np.array([0, 0, 0])
    for i in range(num_points-1):
        pt_not_found = True
        while pt_not_found:
            step = np.random.standard_normal((1, 3))
            magnitude_step = np.linalg.norm(step)
            point = points[i] + np.divide(step, magnitude_step) * step_sizes[i]
            if in_confinement(point, confine_type, confine_length):
                points = np.vstack([points, point])
                pt_not_found = False
    return points


def confined_uniform_random_walk(
    num_points: int,
    step_size: float,
    confine_type: str,
    confine_length: float
) -> np.ndarray:
    """Generate coordinates for uniform random walk w/ fixed path length.

    Parameters
    ----------
    num_points : int
        Number of points in the uniform random walk
    step_size : float
        Distance between each point in the random walk
    confine_type : str
        Name of the confining boundary. To indicate model w/o confinement,
        enter a blank string for this argument
    confine_length : double
        The lengthscale associated with the confining boundary. Length
        representation specified in function associated w/ `confine_type`

    Returns
    -------
    np.ndarray (N, 3)
        Coordinates of each point in the uniform random walk, where rows
        represent individual points and columns give x, y, z coordinates
    """
    points = np.array([0, 0, 0])
    for i in range(num_points-1):
        pt_not_found = True
        while pt_not_found:
            step = np.random.uniform(size=(1, 3))
            magnitude_step = np.linalg.norm(step)
            point = points[i] + np.divide(step, magnitude_step) * step_size
            if in_confinement(point, confine_type, confine_length):
                points = np.vstack([points, point])
                pt_not_found = False
    return points


def looped_confined_gaussian_walk(
    num_points: int,
    step_size: float,
    confine_type: str,
    confine_length: float
) -> np.ndarray:
    """Generate looped, confined Gaussian random walk w/ fixed path length.

    Parameters
    ----------
    num_points : int
        Number of points in the Gaussian random walk
    step_size : float
        Distance between each point in the random walk
    confine_type : str
        Name of the confining boundary. To indicate model w/o confinement,
        enter a blank string for this argument
    confine_length : double
        The lengthscale associated with the confining boundary. Length
        representation specified in function associated w/ `confine_type`

    Returns
    -------
    np.ndarray (N, 3)
        Coordinates of each point in the Gaussian random walk, where rows
        represent individual points and columns give x, y, z coordinates
    """
    origin = np.array([0, 0, 0])
    points = origin.copy()

    for i in range(num_points-1):
        pt_not_found = True
        dist = np.linalg.norm(points[i] - origin)
        remaining_chain = (num_points - i + 1) * step_size

        # Free step
        if remaining_chain > dist or dist == 0:
            while pt_not_found:
                step = np.random.standard_normal((1, 3))
                magnitude_step = np.linalg.norm(step)
                point = points[i] + np.divide(step, magnitude_step) * step_size
                if in_confinement(point, confine_type, confine_length):
                    points = np.vstack([points, point])
                    pt_not_found = False

        # Constrained step
        else:
            step = (-1) * points[i]
            magnitude_step = np.linalg.norm(step)
            point = points[i] + np.divide(step, magnitude_step) * step_size
            points = np.vstack([points, point])

    return points


def in_confinement(point, confine_type, confine_length):
    """Check if a proposed point lies inside a specified confinement.

    Parameters
    ----------
    point : array_like (1, 3)
        Point for which to evaluate position in confinement
    confine_type : str
        Name of the confining boundary. To indicate model without confinement,
        enter a blank string for this argument.
    confine_length : double
        The lengthscale associated with the confining boundary. What the length
        represents is specified in the function associated with `confine_type`.

    Returns
    -------
    bool
        Indicator for whether the point lies in the confinement (True) or
        outside the confinement (False)
    """
    if confine_type == "":
        return True
    elif confine_type == "Spherical":
        return in_spherical_confinement(point, confine_length)
    elif confine_type == "Cubical":
        return in_cubical_confinement(point, confine_length)
    else:
        raise ValueError("Confinement type not found.")


def in_spherical_confinement(point, boundary_radius):
    """Check if a proposed point lies inside a spherical confinement.

    Parameters
    ----------
    point : array_like (1, 3)
        Point for which to evaluate position in confinement.
    boundary_radius : double
        Radial distance of the confining boundary from the origin.

    Returns
    -------
    bool
        Indicator for whether the point lies in the confinement (True) or
        outside the confinement (False)
    """
    if np.linalg.norm(point) <= boundary_radius:
        return True
    return False


def in_cubical_confinement(point, confine_length):
    """Check if a proposed point lies inside a spherical confinement.

    Parameters
    ----------
    point : array_like (1, 3)
        Point for which to evaluate position in confinement.
    confine_length : double
        Edge length of cubical confimement.

    Returns
    -------
    bool
        Indicator for whether the point lies in the confinement (True) or
        outside the confinement (False)
    """
    for i in range(point.shape[1]):
        if np.abs(point[0, i]) > confine_length / 2:
            return False
    return True


def estimate_tangents_from_coordinates(
    coordinates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate t3 and t2 tangent vectors from array of coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Ordered coordinates representing path of a polymer

    Returns
    -------
    np.ndarray (N, 3)
        Matrix representing t3 tangents, where rows represent x, y, z
        components of tangent at each point.
    np.ndarray (N, 3)
        Matrix representing t2 tangents, where rows represent x, y, z
        coordinates of tangent vector at each point
    """
    num_beads = coordinates.shape[0]
    t3 = np.zeros((num_beads, 3))
    t2 = np.zeros((num_beads, 3))

    first_diff = coordinates[1, :] - coordinates[0, :]
    last_diff = coordinates[num_beads-1, :] - coordinates[num_beads-2, :]
    t3[0, :] = first_diff / np.linalg.norm(first_diff)
    t3[num_beads-1, :] = last_diff / np.linalg.norm(last_diff)
    for i in range(1, num_beads-1):
        surrounding_diff = coordinates[i+1, :] - coordinates[i-1, :]
        t3[i, :] = surrounding_diff / np.linalg.norm(surrounding_diff)

    arbitrary_vec_1 = np.array([1, 0, 0])
    arbitrary_vec_2 = np.array([0, 1, 0])
    for i in range(num_beads):
        trial_t2 = np.cross(t3[i, :], arbitrary_vec_1)
        if np.all(trial_t2 == 0):
            trial_t2 = np.cross(t3[i, :], arbitrary_vec_2)
        t2[i, :] = trial_t2 / np.linalg.norm(trial_t2)

    return t3, t2
