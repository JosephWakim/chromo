"""Functions for specification of initial polymer paths.
"""

from typing import Callable, Tuple, Optional, List

import numpy as np


def coordinates_in_x_y(
    num_beads: int,
    bead_length: float,
    shape_func: Callable[[float], float],
    step_size: float
) -> np.ndarray:
    """Find bead coordiantes about path in x, y plane w/ fixed contour length.

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
    shape_func_x, shape_func_y, shape_func_z : Callable[[float], float]
        Parametric functions to obtain the x, y, z coordinates of the path
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
    for i in range(1, num_beads):
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
    shape_func_x, shape_func_y, shape_func_z : Callable[[float], float]
        Parametric functions to obtain the x, y, z coordinates of the path
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
    if r is None:
        r = np.array(
            [
                [shape_func_x(t_i), shape_func_y(t_i), shape_func_z(t_i)]
                for t_i in t
            ]
        )
    num_beads = len(t)
    t3 = np.zeros((num_beads, 3))
    t2 = np.zeros((num_beads, 3))

    arbitrary_vec_1 = np.array([1, 0, 0])
    arbitrary_vec_2 = np.array([0, 1, 0])

    for i in range(num_beads):
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
    step_size: float
) -> np.ndarray:
    """Generate coordinates for Gaussian random walk w/ fixed path length.

    Parameters
    ----------
    num_steps : int
        Number of steps in the Gaussian random walk
    step_size : float
        Distance between each point in the random walk

    Returns
    -------
    np.ndarray (N, 3)
        Coordinates of each point in the Gaussian random walk, where rows
        represent individual points and columns give x, y, z coordinates
    """
    steps = np.random.standard_normal((num_steps, 3))
    magnitude_steps = np.linalg.norm(steps, axis=1)
    return np.cumsum(
        np.divide(steps, magnitude_steps[:, None]) * step_size, axis=0
    )


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
