"""Utility functions for linear algebra calculations.
"""

# Built-in Modules
from typing import Tuple
from random import uniform

# External Modules
import numpy as np


def uniform_sample_unit_sphere() -> Tuple[float, float, float]:
    """Randomly sample a vector on the unit sphere.

    Returns
    -------
    Tuple[float, float, float]
        Random (x, y, z) vector on unit sphere
    """
    phi = np.random.uniform() * (2 * np.pi)
    theta = np.arccos(uniform(-1, 1))
    return np.array([
        np.cos(phi) * np.sin(theta),    # x
        np.sin(phi) * np.sin(theta),    # y
        np.cos(theta)                   # z
    ])


def arbitrary_axis_rotation(axis, point, rot_angle):
    """
    Rotate about an axis defined by two points.

    Generate a transformation matrix for counterclockwise (right handed
    convention) rotation of angle `rot_angle` about an arbitrary axis from
    points `r0` to `r1`.

    Parameters
    ----------
    axis : array_like (3,)
        Rotation axis
    point : array_like (3,)
        Rotation fulcrum
    rot_angle : float
        Angle of rotation. Positive rotation is counterclockwise when
        rotation axis is pointing directly out of the screen

    Returns
    -------
    rot_mat : (4, 4) array_like
        Homogeneous rotation matrix
    """
    rot = np.identity(4)

    rot[0, 0] = axis[0]**2 + (axis[1]**2 + axis[2]**2) *\
        np.cos(rot_angle)
    rot[0, 1] = axis[0] * axis[1] * (1 - np.cos(rot_angle)) -\
        axis[2]*np.sin(rot_angle)
    rot[0, 2] = axis[0] * axis[2] * (1 - np.cos(rot_angle)) +\
        axis[1]*np.sin(rot_angle)

    rot[1, 0] = axis[0] * axis[1] * (1 - np.cos(rot_angle)) +\
        axis[2]*np.sin(rot_angle)
    rot[1, 1] = axis[1]**2 + (axis[0]**2 + axis[2]**2) *\
        np.cos(rot_angle)
    rot[1, 2] = axis[1] * axis[2]*(1 - np.cos(rot_angle)) -\
        axis[0]*np.sin(rot_angle)

    rot[2, 0] = axis[0] * axis[2] * (1 - np.cos(rot_angle)) -\
        axis[1]*np.sin(rot_angle)
    rot[2, 1] = axis[1] * axis[2] * (1 - np.cos(rot_angle)) +\
        axis[0]*np.sin(rot_angle)
    rot[2, 2] = axis[2]**2 + (axis[0]**2 + axis[1]**2) *\
        np.cos(rot_angle)

    rot_vec = np.cross(point, axis) * np.sin(rot_angle)

    rot_vec[0] += (
        point[0] * (1 - axis[0]**2) - axis[0] *
        (point[1] * axis[1] + point[2] * axis[2])
        ) * (1 - np.cos(rot_angle))
    rot_vec[1] += (
        point[1] * (1 - axis[1]**2) - axis[1] *
        (point[0] * axis[0] + point[2] * axis[2])
        ) * (1 - np.cos(rot_angle))
    rot_vec[2] += (
        point[2]*(1 - axis[2]**2) - axis[2] *
        (point[0] * axis[0] + point[1] * axis[1])
        ) * (1 - np.cos(rot_angle))

    rot[0:3, 3] = rot_vec

    return rot


def generate_translation_mat(delta_x, delta_y, delta_z):
    """
    Generate translation matrix.

    Generate the homogeneous transformation matrix for a translation
    of distance delta_x, delta_y, delta_z in the x, y, and z directions,
    respectively.

    Parameters
    ----------
    delta_x : float
        Distance to translate in x-direction
    delta_y : float
        Distance to translate in y-direction
    delta_z : float
        Distance to translate in z-direction

    Returns
    -------
    translation_mat : (4, 4) array_like
        Homogeneous translation matrix
    """

    translation_mat = np.identity(4)
    translation_mat[0, 3] = delta_x
    translation_mat[1, 3] = delta_y
    translation_mat[2, 3] = delta_z

    return translation_mat


def get_prism_verticies(
    num_sides: int,
    width: float,
    height: float
) -> np.ndarray:
    """Get the cartesian coordinates of verticies for specified prism geometry.

    Parameters
    ----------
    num_sides : int
        Number of sides on the face of the prism used to represent the
        nucleosome's geometry; this determines the locations of verticies
        of the `DetailedNucleosome`
    width, height : float
        Determines the shape of the prism defining the location of the
        nucleosome's verticies. The `width` gives the diameter of the
        circle circumscribing the base of the prism. The `height` gives
        the height of the prism.

    Returns
    -------
    np.ndarray (M, 3)
        Cartesian coordinates of the M verticies of the prism geometry.
    """
    ang = 2 * np.pi / num_sides
    base_coords_2D = np.array([1, 0])
    rot_mat = np.array(
        [[np.cos(ang), -np.sin(ang)],
         [np.sin(ang), np.cos(ang)]]
    )
    for i in range(1, num_sides):
        new_coords = rot_mat @ base_coords_2D[i-1]
        base_coords_2D = np.stack(base_coords_2D, new_coords)
    base_coords_2D *= width / 2

    verticies = np.zeros((num_sides * 2, 3))
    for i in range(num_sides):
        verticies[i, 0:2] = base_coords_2D[i]
        verticies[i, 2] = -height / 2
        verticies[2 * i, 0:2] = base_coords_2D[i]
        verticies[2 * i, 2] = height / 2
    return verticies
