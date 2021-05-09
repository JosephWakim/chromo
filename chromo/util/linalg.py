"""Utility functions for linear algebra calculations.
"""

# Built-in Modules
from typing import Tuple
from random import uniform

# External Modules
import numpy as np
from scipy.spatial.transform import Rotation


def uniform_sample_unit_sphere():
    """Randomly sample a vector on the unit sphere.

    Returns
    -------
    Tuple[float, float, float]
        Random (x, y, z) vector on unit sphere
    """
    phi = np.random.uniform() * (2 * np.pi)
    theta = np.arccos(uniform(-1, 1))
    return (
        np.cos(phi) * np.sin(theta),    # x
        np.sin(phi) * np.sin(theta),    # y
        np.cos(theta)                   # z
    )


def arbitrary_axis_rotation(r0, r1, rot_angle):
    """
    Rotate about an axis defined by two points.

    Generate a transformation matrix for counterclockwise (right handed
    convention) rotation of angle `rot_angle` about an arbitrary axis from
    points `r0` to `r1`.

    Parameters
    ----------
    r0 : (3,) array_like
        First point defining axis to rotate around.
    r1 : (3,) array_like
        Second point defining axis to rotate around.
    rot_angle : float
        Angle of rotation. Positive rotation is counterclockwise when the
        vector `r1 - r0` is pointing directly out of the screen.

    Returns
    -------
    rot_mat : (4, 4) array_like
        Homogeneous rotation matrix.
    """
    translate_mat = np.array([
        [1, 0, 0, -r0[0]],
        [0, 1, 0, -r0[1]],
        [0, 0, 1, -r0[2]],
        [0, 0, 0, 1]
    ])
    inv_translation_mat = translate_mat.copy()
    inv_translation_mat[0:3, 3] *= -1

    rot_axis = r1 - r0
    rot_vec = rot_angle * rot_axis/np.linalg.norm(rot_axis)
    rot_mat = np.identity(4)
    rot_mat[:3, :3] = Rotation.from_rotvec(rot_vec).as_matrix()

    return inv_translation_mat @ rot_mat @ translate_mat


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
