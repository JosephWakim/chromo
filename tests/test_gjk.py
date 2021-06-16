"""Test the GJK Algorithm.

Unit tests translated from FORTRAN code prepared by Nicole Pagane published at:
https://github.com/SpakowitzLab/wlcsim/blob/master/src/util/sterics.f90
"""

import numpy as np

from chromo.util.gjk import gjk_collision


def test_same_shape():
    """Test that the GJK Algorithm detects two equivalent shapes.
    """
    vertices_1 = np.array(
        [
            [4.76313972, 0, -4.76313972, -4.76313972, 0, 4.76313972,
             4.76313972, 0, -4.76313972, -4.76313972, 0, 4.76313972],
            [2.75, 5.5, 2.75, -2.75, -5.5, -2.75,
             2.75, 5.5, 2.75, -2.75, -5.5, -2.75],
            [0, 0, 0, 0, 0, 0, -3, -3, -3, -3, -3, -3]
        ]
    ).T
    vertices_2 = vertices_1.copy()
    max_iters = 1000

    assert gjk_collision(vertices_1, vertices_2, max_iters)


def test_no_intersection_x():
    """Test no intersection between neighboring geometries in x.
    """
    vertices_1 = np.array(
        [
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5],
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [4.5, 4, 4.5, 5.5, 6, 5.5, 4.5, 4, 4.5, 5.5, 6, 5.5],
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    max_iters = 1000

    assert not gjk_collision(vertices_1, vertices_2, max_iters)


def test_intersection_x():
    """Test intersection between geometries in x.
    """
    vertices_1 = np.array(
        [
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5],
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [0.5, 0, 0.5, 1.5, 2, 1.5, 0.5, 0, 0.5, 1.5, 2, 1.5],
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    max_iters = 1000

    assert gjk_collision(vertices_1, vertices_2, max_iters)


def test_tangent_x():
    """Test tangentially aligned objects in x.
    """
    vertices_1 = np.array(
        [
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5],
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [1.5, 1, 1.5, 2.5, 3, 2.5, 1.5, 1, 1.5, 2.5, 3, 2.5],
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    max_iters = 1000

    assert gjk_collision(vertices_1, vertices_2, max_iters)


def test_no_intersection_y():
    """Test no intersection between neighboring geometries in y.
    """
    vertices_1 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [4.5, 4, 4.5, 5.5, 6, 5.5, 4.5, 4, 4.5, 5.5, 6, 5.5],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    max_iters = 1000

    assert not gjk_collision(vertices_1, vertices_2, max_iters)


def test_intersection_y():
    """Test intersection between geometries in y.
    """
    vertices_1 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0.5, 0, 0.5, 1.5, 2, 1.5, 0.5, 0, 0.5, 1.5, 2, 1.5],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    max_iters = 1000

    assert gjk_collision(vertices_1, vertices_2, max_iters)


def test_tangent_y():
    """Test tangentially aligned objects in y.
    """
    vertices_1 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [1.5, 1, 1.5, 2.5, 3, 2.5, 1.5, 1, 1.5, 2.5, 3, 2.5],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        ]
    ).T
    max_iters = 1000

    assert gjk_collision(vertices_1, vertices_2, max_iters)


def test_no_intersection_z():
    """Test no intersection between neighboring geometries in z.
    """
    vertices_1 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
            [4.5, 4, 4.5, 5.5, 6, 5.5, 4.5, 4, 4.5, 5.5, 6, 5.5]
        ]
    ).T
    max_iters = 1000

    assert not gjk_collision(vertices_1, vertices_2, max_iters)


def test_intersection_z():
    """Test intersection between geometries in z.
    """
    vertices_1 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
            [0.5, 0, 0.5, 1.5, 2, 1.5, 0.5, 0, 0.5, 1.5, 2, 1.5]
        ]
    ).T
    max_iters = 1000

    assert gjk_collision(vertices_1, vertices_2, max_iters)


def test_tangent_z():
    """Test tangentially aligned objects in z.
    """
    vertices_1 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
            [-0.5, -1, -0.5, 0.5, 1, 0.5, -0.5, -1, -0.5, 0.5, 1, 0.5]
        ]
    ).T
    vertices_2 = np.array(
        [
            [-0.8660, 0, 0.8660, 0.8660, 0, -0.8660,
             -0.8660, 0, 0.8660, 0.8660, 0, -0.8660],
            [0, 0, 0, 0, 0, 0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
            [1.5, 1, 1.5, 2.5, 3, 2.5, 1.5, 1, 1.5, 2.5, 3, 2.5]
        ]
    ).T
    max_iters = 1000

    assert gjk_collision(vertices_1, vertices_2, max_iters)
