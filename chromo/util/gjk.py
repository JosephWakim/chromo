"""Gilbert-Johnson-Keerthi (GJK) Collision Detection

Adapted from matlab function provided by Matthew Sheen (2016) at:
https://github.com/mws262/MATLAB-GJK-Collision-Detection/blob/master/GJK.m

In our implementation of the GJK algorithm, point c will be the oldest support
point selected, followed by point b, and lastly point a. This way, between
iterations, point a will become point b and point b will become point c.
"""
from typing import Tuple

import numpy as np


def gjk_collision(
    verticies_1: np.ndarray,
    verticies_2: np.ndarray,
    max_iters: int
) -> bool:
    """Test for collision by GJK algorithm given arrays of vertex coordinates.

    Parameters
    ----------
    verticies_1, verticies_2 : np.ndarray (M, 3)
        Arrays of verticies corresponding to two objects, where the rows
        represent individual vertices and columns represent cartesian (x, y, z)
        coordinates for the corresponding vertices.
    max_iters : int
        Maximum iterations of the GJK algorithm to try

    Returns
    -------
    bool
        Flag for collision between objects (True = collision, False = no
        collision)
    """
    if shape_overlap(verticies_1, verticies_2):
        return True

    center_1 = np.average(verticies_1, axis=0)
    center_2 = np.average(verticies_2, axis=0)

    diff = center_2 - center_1
    dir_ = diff / np.linalg.norm(diff)

    a, b = pick_line(verticies_1, verticies_2, dir_)
    a, b, c, flag = pick_triangle(a, b, verticies_1, verticies_2, max_iters)

    # Check for overlapping tetrahedrons if a triangle contains origin
    if flag:
        return pick_tetrahedron(c, b, a, verticies_1, verticies_2, max_iters)

    return False


def shape_overlap(s1: np.ndarray, s2: np.ndarray) -> bool:
    """Check if any two vertices of the two shapes overlap

    Parameters
    ----------
    s1, s2 : np.ndarray (M, 3)
        Verticies of the two shapes

    Returns
    -------
    bool
        Flag indicating overlapping verticies (True = overlapping verticies ->
        collision, False = non-overlapping verticies -> more analysis needed)
    """
    for i in s1:
        for j in s2:
            ij = i - j
            if np.all(np.isclose(ij, 0)):
                return True
    else:
        return False


def pick_line(
    vertices_1: np.ndarray,
    vertices_2: np.ndarray,
    dir_: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick an initial line along the Minkowski difference between the two shapes.

    Parameters
    ----------
    verticies_1, verticies_2 : np.ndarray (M, 3)
        Arrays of verticies corresponding to two objects, where the rows
        represent individual vertices and columns represent cartesian (x, y, z)
        coordinates for the corresponding vertices.
    dir_ : np.ndarray (3, )
        Initial direction with which to select support points

    Returns
    -------
    np.ndarray (2, 3)
        Two support points as Cartesian (x, y, z) coordinates
    """
    a = support_function(vertices_1, vertices_2, dir_)
    b = support_function(vertices_1, vertices_2, -dir_)
    return a, b


def support_function(vertices_1, vertices_2, dir_):
    """Support function for obtaining support points on Minkowski difference.

    Parameters
    ----------
    verticies_1, verticies_2 : np.ndarray (M, 3)
        Arrays of verticies corresponding to two objects, where the rows
        represent individual vertices and columns represent cartesian (x, y, z)
        coordinates for the corresponding vertices.
    dir_ : np.ndarray (3, )
        Initial direction with which to select support points

    Returns
    -------
    np.ndarray (3, )
        Support point in given direction given vertex
    """
    point_1 = get_furthest_in_dir(vertices_1, dir_)
    point_2 = get_furthest_in_dir(vertices_2, -dir_)
    return point_1 - point_2


def get_furthest_in_dir(vertices: np.ndarray, dir_: np.ndarray) -> np.ndarray:
    """Get the furthest point of a shape from its center in a given direction.

    Parameters
    ----------
    verticies : np.ndarray (M, 3)
        Verticies of shape
    dir_ : np.ndarray (3, )
        Direction in which to get the furthest point

    Returns
    -------
    np.ndarray
        Cartesian (x, y, z) coordinates of the furthest point in the specified
        direction.
    """
    dots = np.dot(vertices, dir_)
    furthest_ind = np.argmax(dots)
    return vertices[furthest_ind]


def pick_triangle(
    a: np.ndarray,
    b: np.ndarray,
    verticies_1: np.ndarray,
    verticies_2: np.ndarray,
    max_iters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Check if 2D simplex (triangle) contains origin.

    If two convex shapes collide, then there exists a triangle formed from
    vertices of the Minkowski difference which contains the origin.

    Begin by initializing a flag indicating whether the generated triangle
    contains the origin – initialize as false.

    Generate the first support point using the first two points specified
    as arguments. This support point must be selected in the direction
    towards the origin, which is obtained using a triple product.

    Any triangle can be used to partition 2D space into seven Voronoi regions:
    a, b, c, ab, ac, bc, abc. If our triangle contains the origin, then the
    orgin is located in region abc. Because of the way we selected our three
    points of our triangle, we can eliminate certain regions as candidates for
    containing the origin – we eliminate regions a, b, c, bc. If our triangle
    contains the origin, then regions ab and ac do NOT contain the origin; to
    check that our triangle contains the origin, verify that this is the case.

    If region ab contain the origin, then the dot product of abp (perpendicular
    to side ab towards the origin) and ao (the vector from the origin to a)
    will be positive. Likewise, if region ac contains the origin, then the dot
    product between acp (perpendicular to side ac towards the origin) and ao
    will be positive.

    Parameters
    ----------
    a, b : np.ndarray (3, )
        Coordinates of the older support point (b) and newer support point (a)
    verticies_1, verticies_2 : np.ndarray (M, 3)
        Arrays of verticies corresponding to two objects, where the rows
        represent individual vertices and columns represent cartesian (x, y, z)
        coordinates for the corresponding vertices.
    max_iters : int
        Maximum iterations of the GJK algorithm to try

    Returns
    -------
    np.ndarray (3, ) x3
        Verticies of new support points on triangle
    bool
        Flag for triangle containing origin (True = contains origin, False =
        does not contain origin)
    """
    contains_origin = False

    ab = b - a
    ao = -a
    dir_ = np.cross(np.cross(ab, ao), ab)

    c = b
    b = a
    a = support_function(verticies_1, verticies_2, dir_)    # make 1st triangle

    for i in range(max_iters):
        ab = b - a
        ao = -a
        ac = c - a
        abc = np.cross(ab, ac)

        abp = np.cross(ab, abc)
        acp = np.cross(abc, ac)

        if np.dot(abp, ao) > 0:     # check if origin in region ab
            c = b
            b = a
            dir_ = abp
        elif np.dot(acp, ao) > 0:   # check if origin in region ac
            b = a
            dir_ = acp
        else:                       # origin must be in region abc!
            contains_origin = True
            break
        a = support_function(verticies_1, verticies_2, dir_)

    return a, b, c, contains_origin


def pick_tetrahedron(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    verticies_1: np.ndarray,
    verticies_2: np.ndarray,
    max_iters: int
) -> bool:
    """Check if 3D simplex (tetrahedron) contains origin, indicating collision.

    If two 3D, convex shapes collide, then a 3D simplex can be drawn between
    the Minkowski difference of the two shapes which contains the origin. This
    function only needs to be evaluated if a triangle case contains the origin.

    Parameters
    ----------
    a, b, c : np.ndarray (3, )
        Coordinates of the oldest support point (c), second oldest support
        point (b), and newest support point (a) at the current iteration of
        the GJK algorithm.
    verticies_1, verticies_2 : np.ndarray (M, 3)
        Arrays of verticies corresponding to two objects, where the rows
        represent individual vertices and columns represent cartesian (x, y, z)
        coordinates for the corresponding vertices.
    max_iters : int
        Maximum iterations of the GJK algorithm to try

    Returns
    -------
    bool
        Flag for tetrahedron containing origin (True = contains origin, False =
        does not contain origin) -> indicator of whether 3D collision occured
    """
    contains_origin = False

    ab = b - a
    ac = c - a
    abc_p = np.cross(ab, ac)
    ao = -a

    if np.dot(abc_p, ao) > 0:   # Origin lies above triangle abc
        d = c
        c = b
        b = a
        dir_ = abc_p
        a = support_function(verticies_1, verticies_2, dir_)
    else:                       # Origin lies below triangle abc
        d = b
        b = a
        dir_ = -abc_p
        a = support_function(verticies_1, verticies_2, dir_)

    for i in range(max_iters):
        ab = b - a
        ao = -a
        ac = c - a
        ad = d - a

        abc = np.cross(ab, ac)

        # Use above abc case as default; adjust if origin below triangle
        if np.dot(abc, ao) <= 0:
            acd_p = np.cross(ac, ad)        # perpendicular to face acd
            if np.dot(acd_p, ao) > 0:       # origin lies above triangle acd
                b = c
                c = d
                ab = ac
                ac = ad
                abc_p = acd_p
            elif np.dot(acd_p, ao) < 0:     # origin lies below triangle acd
                adb_p = np.cross(ad, ab)

                if np.dot(adb_p, ao) < 0:   # origin lies above triangle adb
                    c = b
                    b = d
                    ac = ab
                    ab = ad
                    abc_p = adb_p
                else:   # origin does not lie above triangle adb
                    # origin must be in tetrahedron
                    contains_origin = True
                    break

        if np.dot(abc, ao) > 0:     # Origin lies above new triangle abc
            d = c
            c = b
            b = a
            dir_ = abc_p
            a = support_function(verticies_1, verticies_2, dir_)
        else:                       # Origin lies below new triangle abc
            d = b
            b = a
            dir_ = -abc_p
            a = support_function(verticies_1, verticies_2, dir_)

    return contains_origin
