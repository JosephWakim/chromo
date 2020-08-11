
"""
Utility functions for linear algebra calculations

"""

import random
import numpy as np
import math as math


def arbitrary_axis_rotation(r_ind0, r_ind1, rot_angle):
    """
    Generate a transformation matrix for rotation of angle rot_angle about an arbitrary axis from points r_ind0 to r_ind1.

    Parameters
    ----------
    r_ind0:         (3, 1) np.array
                    1D column vector of (x, y, z) coordinates for the first point forming the axis of rotation

    r_ind1:         (3, 1) np.array
                    1D column vector of (x, y, z) coordinates for the second point forming the axis of rotation

    rot_angle:      float
                    Magnitude of the angle of rotation about arbitrary axis

    Returns
    -------
    rot_matrix:     (4, 4) np.array
                    Homogeneous rotation matrix for rotation about arbitrary axis

    """

    # Generate translation matrix such that neighboring point is translated to origin
    translate_mat = np.zeros((4,4))
    translate_mat[0, 0] = 1
    translate_mat[1, 1] = 1
    translate_mat[2, 2] = 1
    translate_mat[3, 3] = 1
    translate_mat[0, 3] = -r_ind1[0]
    translate_mat[1, 3] = -r_ind1[1]
    translate_mat[2, 3] = -r_ind1[2]

    # Generate the inverse of the translation mat
    inv_translation_mat = translation_mat
    inv_translation_mat[0, 3] = -translation_mat[0, 3]
    inv_translation_mat[1, 3] = -translation_mat[1, 3]
    inv_translation_mat[2, 3] = -translation_mat[2, 3]

    # Calculate the length of the projections to point ind0 on yz plane.
    proj_len_yz = math.sqrt(r_ind0[2]**2 + r_ind0[1]**2)

    # Generate rotation matrix such that origin to ind0 is rotated onto the xz plane
    rot_mat_x = np.zeros((4,4))
    rot_mat_x[0, 0] = 1
    rot_mat_x[1, 1] = r_ind0[2] / proj_len_yz
    rot_mat_x[1, 2] = -r_ind0[1] / proj_len_yz
    rot_mat_x[2, 1] = r_ind0[1] / proj_len_yz
    rot_mat_x[2, 2] = r_ind0[2] / proj_len_yz
    rot_mat_x[3, 3] = 1

    # Generate the inverse of the x rotation matrix
    inv_rot_mat_x = rot_mat_x
    inv_rot_mat_x[1, 2] = -rot_mat_x[1, 2]
    inv_rot_mat_x[2, 1] = -rot_mat_x[2, 1]

    # Generate rotation matrix around the y-axis such that the origin to ind0 is rotated onto the z-axis
    rot_mat_y = np.zeros((4,4))
    rot_mat_y[0, 0] = proj_len_yz
    rot_mat_y[0, 2] = -r_ind0[0]
    rot_mat_y[1, 1] = 1
    rot_mat_y[2, 0] = r_ind0[0]
    rot_mat_y[2, 2] = proj_len_yz
    rot_mat_y[3, 3] = 1

    # Generate the inverse of the y rotation matrix
    inv_rot_mat_y = rot_mat_y
    inv_rot_mat_y[0, 2] = -rot_mat_y[0, 2]
    inv_rot_mat_y[2, 0] = -rot_mat_y[2, 0]

    # Generate rotation matrix about the z-axis using the specified rotation angle.
    rot_mat_z = np.zeros((4, 4))
    rot_mat_z[0, 0] = math.cos(rot_angle)
    rot_mat_z[0, 1] = -math.sin(rot_angle)
    rot_mat_z[1, 0] = -rot_mat_z[0, 1]
    rot_mat_z[1, 1] = rot_mat_z[0, 0]
    rot_mat_z[2, 2] = 1
    rot_mat_z[3, 3] = 1

    # Generate full rotation matrix
    rot_matrix = np.matmul(inv_translation_mat, \
        np.matmul(inv_rot_mat_x, \
            np.matmul(inv_rot_mat_y, \
                np.matmul(rot_mat_z, \
                    np.matmul(rot_mat_y, \
                        np.matmul(rot_mat_x, translate_mat))))))

    return(rot_matrix)

