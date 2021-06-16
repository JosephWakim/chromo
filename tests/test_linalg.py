import chromo.util.linalg as linalg

import numpy as np


def test_z_axis_rotation():
    """Test 45 degree rotation about the z-axis.
    """
    r0 = np.array([0, 0, 0])
    r1 = np.array([0, 0, 1])
    axis = r1 - r0
    point = np.array([0, 0, 0])
    rot_angle = np.pi / 4

    rot_mat = linalg.arbitrary_axis_rotation(axis, point, rot_angle)
    rot_mat_inv_inv = np.linalg.inv(
        linalg.arbitrary_axis_rotation(r1, r0, -rot_angle)
    )

    expected_rot_mat = np.identity(4)
    expected_rot_mat[0:2, 0:2] = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)]
    ])

    assert np.all(np.isclose(rot_mat, rot_mat_inv_inv))
    assert np.all(np.isclose(rot_mat, expected_rot_mat))
    assert np.all(np.isclose(rot_mat_inv_inv, expected_rot_mat))


def test_translation():
    """Test translation matrix.
    """
    r0 = np.ones(4)

    delta_x = 0.1
    delta_y = 0.2
    delta_z = 0.3
    translation_mat = linalg.generate_translation_mat(
        delta_x, delta_y, delta_z
    )

    expected_translation_mat = np.array(
        [
            [1, 0, 0, 0.1],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ]
    )

    assert(np.all(np.isclose(translation_mat, expected_translation_mat)))

    r1 = translation_mat @ r0.T
    r1 = r1.T
    r1_expected = np.array([1.1, 1.2, 1.3, 1])

    assert np.all(np.isclose(r1, r1_expected))
