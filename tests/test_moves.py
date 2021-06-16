"""Test the "deterministic" part of each function for proposing MC moves.
"""

# External Modules
import numpy as np

# Custom Modules
import chromo.mc.moves as mv


def test_determinitic_end_pivot():
    """Test End pivot move w/ 90 deg clockwise rotation about positive x axis.

    A 90 degree clockwise rotation is equivalent to a -90 degree rotation about
    the x-axis or a 90 degree rotation about the -x axis.
    """
    axis = np.array([1, 0, 0])
    rot_angle = - np.pi / 2
    c = np.sqrt(0.5)

    r_points = np.array(
        [
            [1,  0, 0, 1],
            [2,  0, 0, 1],
            [3,  0, 0, 1],
            [3, -1, 0, 1],
            [3, -2, 0, 1]
        ]
    ).T

    r_expected = np.array(
        [
            [1, 0, 0, 1],
            [2, 0, 0, 1],
            [3, 0, 0, 1],
            [3, 0, 1, 1],
            [3, 0, 2, 1]
        ]
    ).T

    t3_points = np.array(
        [
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [c, -c, 0, 0],
            [0, -1, 0, 0],
            [0, -1, 0, 0]
        ]
    ).T

    t3_expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [c, 0, c, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ]
    ).T

    t2_points = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ]
    ).T

    t2_expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ]
    ).T

    r, t3, t2 = mv.conduct_end_pivot(
        r_points,
        t3_points,
        t2_points,
        axis,
        r_points[0:3, 0],
        rot_angle
    )

    assert np.all(np.isclose(r, r_expected))
    assert np.all(np.isclose(t3, t3_expected))
    assert np.all(np.isclose(t2, t2_expected))


def test_determinitic_slide_move():
    """Test deterministic component of slide move.
    """

    r_points = np.array(
        [
            [1,  0, 0, 1],
            [2,  0, 0, 1],
            [3,  0, 0, 1],
            [3, -1, 0, 1],
            [3, -2, 0, 1]
        ]
    ).T

    translate_x = 1
    translate_y = 2
    translate_z = 3.5

    r_expected = np.array(
        [
            [2, 2, 3.5, 1],
            [3, 2, 3.5, 1],
            [4, 2, 3.5, 1],
            [4, 1, 3.5, 1],
            [4, 0, 3.5, 1]]
        ).T

    r_observed = mv.conduct_slide(
        r_points,
        translate_x,
        translate_y,
        translate_z
    )

    assert np.all(np.isclose(r_observed, r_expected))


def test_deterministic_tangent_rotation():
    """Test the deterministic component of the tangent rotation move.

    Rotate the tangent vectors (2 * pi / 3) degrees counterclockwise about an
    axis defined by [1, 1, 1] / sqrt(3).
    """
    t3_point = np.array([0, 0, 1, 0])
    t2_point = np.array([0, 1, 0, 0])

    axis = np.array([1, 1, 1]) / np.sqrt(3)
    rot_angle = 2 * np.pi / 3

    t3_expected = np.array([1, 0, 0, 0])
    t2_expected = np.array([0, 0, 1, 0])

    t3_observed, t2_observed = mv.conduct_tangent_rotation(
        t3_point, t2_point, axis, rot_angle
    )

    assert np.all(np.isclose(t3_observed, t3_expected))
    assert np.all(np.isclose(t2_observed, t2_expected))


def test_deterministic_crank_shaft_move():
    """Test deterministic component of the crank-shaft move.
    """
    rot_angle = np.pi / 2
    c = np.sqrt(0.5)

    r_poly = np.array(
        [
            [1,  0, 0, 1],
            [2,  0, 0, 1],
            [3,  0, 0, 1],
            [3, -1, 0, 1],  # 3
            [3, -2, 0, 1],  # 4
            [4, -2, 0, 1],  # 5
            [5, -2, 0, 1],  # 6
            [5, -1, 0, 1],  # 7
            [5,  0, 0, 1],
            [6,  0, 0, 1],
            [7,  0, 0, 1]
        ]
    )

    r_points = r_poly[3:8, :].T

    r_expected = np.array(
        [
            [3, 0, 1, 1],   # 3
            [3, 0, 2, 1],   # 4
            [4, 0, 2, 1],   # 5
            [5, 0, 2, 1],   # 6
            [5, 0, 1, 1]    # 7
        ]
    ).T

    t3_poly = np.array(
        [
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [c, -c, 0, 0],
            [0, -1, 0, 0],  # 3
            [1,  0, 0, 0],  # 4
            [1,  0, 0, 0],  # 5
            [-c, c, 0, 0],  # 6
            [0,  1, 0, 0],  # 7
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [1,  0, 0, 0]
        ]
    )

    t3_points = t3_poly[3:8, :].T

    t3_expected = np.array(
        [
            [0,  0, 1,  0],  # 3
            [1,  0, 0,  0],  # 4
            [1,  0, 0,  0],  # 5
            [-c, 0, -c, 0],  # 6
            [0,  0, -1, 0]   # 7
        ]
    ).T

    axis = r_poly[2, 0:3] - r_poly[8, 0:3]
    axis = axis / np.linalg.norm(axis)

    r, t3 = mv.conduct_crank_shaft(
        r_points,
        t3_points,
        axis,
        rot_angle
    )

    assert np.all(np.isclose(r, r_expected))
    assert np.all(np.isclose(t3, t3_expected))
