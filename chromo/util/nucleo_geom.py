"""Implementation of nucleosome geometry.

By:     Joseph Wakim
Date:   January 24, 2024
Group:  Spakowitz Lab

This module contains functions for computing entry/exit positions and
orientations for DNA around nucleosomes. The Nucleosome is defined by
a reference orientation, such that the t3 axis aligns with the DNA entering
the nucleosome.

We include functions that generate the t3 tangent, normal vector, and binormal
vectors on a local nucleosome reference frame based on the amount of DNA
wrapping. The t2 and t1 tangents are orthogonal to the t3 tangent and are
therefore defined as a linear combination of the normal and binormal vectors.
"""

from typing import Optional
from types import MappingProxyType
import numpy as np

# Twist density of nucleosome-bound DNA (bp / full (2pi) rotation)
NUC_TWIST_DENS = 10.17
# nm length per base pair
LENGTH_BP = 0.332

# Radius of a nucleosome
R_default = 4.1899999999999995
# Rise in DNA path with each loop around the nucleosome
h_default = 4.531142964071856 / 2
# Number of base pairs wrapped around each nucleosome
bp_wrap_default = 147
# nm length of DNA wrapped around each nucleosome
s_default = (bp_wrap_default - 1) * LENGTH_BP
# Natural twist of DNA (radians per bp)
w0_default = 2 * np.pi / (NUC_TWIST_DENS * LENGTH_BP)
# DNA length required for full (2pi) rotation around nucleosome
Lt_default = np.sqrt(4 * np.pi**2 * R_default**2 + h_default**2)
# Natural twist of wrapped DNA, correcting for helical torsion
Phi_default = w0_default - 2 * np.pi * h_default / (Lt_default**2)
# Collection of geometric constants
consts_dict = MappingProxyType({
    "R": R_default, "h": h_default, "bp_wrap": bp_wrap_default,
    "length_bp": LENGTH_BP, "s": s_default, "nuc_twist_dens": NUC_TWIST_DENS,
    "w0": w0_default, "Lt": Lt_default, "Phi": Phi_default
})


def t3(s, consts=consts_dict):
    """Get the t3 orientation in the local reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    np.ndarray (3,) of float
        t3 orientation of DNA at linear position L_wrap in local
        reference frame
    """
    R = consts["R"]
    Lt = consts["Lt"]
    h = consts["h"]
    return np.array([
        - 2 * np.pi * R / Lt * np.sin(2 * np.pi * s / Lt),
        2 * np.pi * R / Lt * np.cos(2 * np.pi * s / Lt),
        h / Lt
    ])


def normal(s, consts=consts_dict):
    """Get the normal vector in the local reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    np.ndarray (3,) of float
        Normal vector of exiting DNA in local reference frame;
        orthogonal to the exiting t3 vector in the local
        reference frame.
    """
    Lt = consts["Lt"]
    return np.array([
        -np.cos(2 * np.pi * s / Lt),
        -np.sin(2 * np.pi * s / Lt),
        0
    ])


def binormal(s, consts=consts_dict):
    """Get the binormal vector in the local reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    np.ndarray (3,) of float
        Binormal vector of exiting DNA in local reference frame;
        orthogonal to the exiting t3 vector and the normal vector
        in the local reference frame.
    """
    return np.cross(t3(s, consts), normal(s, consts))


def t1(s, consts=consts_dict):
    """Get the t1 orientation in the local reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    np.ndarray (3,) of float
        t1 orientation of DNA at linear position s in local
        reference frame
    """
    Phi = consts["Phi"]
    return (
            np.cos(Phi * s) * normal(s, consts) +
            np.sin(Phi * s) * binormal(s, consts)
    )


def t2(s, consts=consts_dict):
    """Get the t2 orientation in the local reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    np.ndarray (3,) of float
        t2 orientation of DNA at linear position s in local
        reference frame
    """
    Phi = consts["Phi"]
    return (
            -np.sin(Phi * s) * normal(s, consts) +
            np.cos(Phi * s) * binormal(s, consts)
    )


def get_T3(s, T3_enter, T2_enter, T1_enter, consts=consts_dict):
    """Get the exiting t3 orientation in the GLOABL reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    T3_enter, T2_enter, T1_enter : np.ndarray (3,) of float
        t3, t2, t1 orientation vectors of entering DNA in GLOBAL
        reference frame
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    np.ndarray (3,) of float
        t3 orientation of DNA at linear position s in GLOBAL
        reference frame
    """
    return (
        np.dot(t3(s=s, consts=consts), t3(s=0, consts=consts)) * T3_enter +
        np.dot(t3(s=s, consts=consts), t2(s=0, consts=consts)) * T2_enter +
        np.dot(t3(s=s, consts=consts), t1(s=0, consts=consts)) * T1_enter
    )


def get_T1(s, T3_enter, T2_enter, T1_enter, consts=consts_dict):
    """Get the exiting t1 orientation in the GLOABL reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    T3_enter, T2_enter, T1_enter : np.ndarray (3,) of float
        t3, t2, t1 orientation vectors of entering DNA in GLOBAL
        reference frame
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    np.ndarray (3,) of float
        t1 orientation of DNA at linear position s in GLOBAL
        reference frame
    """
    return (
        np.dot(t1(s=s, consts=consts), t3(s=0, consts=consts)) * T3_enter +
        np.dot(t1(s=s, consts=consts), t2(s=0, consts=consts)) * T2_enter +
        np.dot(t1(s=s, consts=consts), t1(s=0, consts=consts)) * T1_enter
    )


def get_T2(T3_exit, T1_exit):
    """Get the exiting t2 orientation in the GLOBAL reference frame.

    Notes
    -----
    The exiting T2 orientation is fixed by the exiting T3 and T1
    orientations, since all orientations must be orthogonal.

    Parameters
    ----------
    T3_exit, T1_exit : np.ndarray (3,) of float
        Exiting t3 and t1 orientations in the GLOBAL reference frame

    Returns
    -------
    np.ndarray (3,) of float
        t2 orientation of DNA at linear position s in GLOBAL
        reference frame
    """
    return np.cross(T3_exit, T1_exit)


def get_exiting_orientations(
    s, T3_enter, T2_enter, T1_enter, consts=consts_dict
):
    """Get all exiting orientations in the GLOABL reference frame.

    Parameters
    ----------
    s : float
        Length of wrapped DNA (in nm)
    T3_enter, T2_enter, T1_enter : np.ndarray (3,) of float
        t3, t2, t1 orientation vectors of entering DNA in GLOBAL
        reference frame
    consts : Optional[Dict[str, float]]
        Dictionary of geometric constants governing nucleosome
        wrapping (see above)

    Returns
    -------
    3 x np.ndarray (3,) of float
        t3, t2, and t1 orientations of DNA at linear position s in GLOBAL
        reference frame (returned in the order: T3_exit, T2_exit, T1_exit)
    """
    T3_exit = get_T3(s, T3_enter, T2_enter, T1_enter, consts)
    T1_exit = get_T1(s, T3_enter, T2_enter, T1_enter, consts)
    T2_exit = get_T2(T3_exit, T1_exit)
    return T3_exit, T2_exit, T1_exit


def get_r(s, consts=consts_dict):
    """Specify the position of the DNA path at segment length s.

    Notes
    -----
    This function gives the path of the DNA on a local reference
    frame. To obtain the DNA position at segment length s on a
    global reference frame, you will need to account for (1) the
    change in nucleosome orientation and (2) the change in
    nucleosome position. Conversion to a global reference frame is
    not done by this function.
    """
    h = consts["h"]
    Lt = consts["Lt"]
    R = consts["R"]
    return np.array([
        R * np.cos(2 * np.pi * s / Lt),
        R * np.sin(2 * np.pi * s / Lt),
        h * s / Lt - ((s_default / Lt * h) / 2)
    ])


# For default conditions, we can pre-compute the rotation (dot product) matrix
default_rot_matrix = np.array([
    [
        np.dot(t1(s=s_default, consts=consts_dict),t1(s=0, consts=consts_dict)),
        np.dot(t1(s=s_default, consts=consts_dict),t2(s=0, consts=consts_dict)),
        np.dot(t1(s=s_default, consts=consts_dict),t3(s=0, consts=consts_dict)),
    ],
    [
        np.dot(t2(s=s_default, consts=consts_dict),t1(s=0, consts=consts_dict)),
        np.dot(t2(s=s_default, consts=consts_dict),t2(s=0, consts=consts_dict)),
        np.dot(t2(s=s_default, consts=consts_dict),t3(s=0, consts=consts_dict)),
    ],
    [
        np.dot(t3(s=s_default, consts=consts_dict),t1(s=0, consts=consts_dict)),
        np.dot(t3(s=s_default, consts=consts_dict),t2(s=0, consts=consts_dict)),
        np.dot(t3(s=s_default, consts=consts_dict),t3(s=0, consts=consts_dict)),
    ]
])
# Given an entry triad, we can compute the exit triad by taking the dot product
# of this rotation matrix with a matrix of the triad.


def compute_exit_orientations_with_default_wrapping(
    T3_enter, T2_enter, T1_enter
):
    """Compute exit orientations with the default nucleosome geometry

    Parameters
    ----------
    T3_enter, T2_enter, T1_enter : np.ndarray (3,) of float
        t3, t2, t1 orientation vectors of entering DNA in GLOBAL
        reference frame

    Returns
    -------
    3 x np.ndarray (3,) of float
        t3, t2, and t1 orientations of DNA at linear position s in GLOBAL
        reference frame (returned in the order: T3_exit, T2_exit, T1_exit)
    """
    T_in_matrix = np.column_stack((T1_enter, T2_enter, T3_enter))
    T_out_matrix = np.dot(T_in_matrix, default_rot_matrix[:3].T)
    T1_exit = T_out_matrix[:, 0]
    T2_exit = T_out_matrix[:, 1]
    T3_exit = T_out_matrix[:, 2]
    return T3_exit, T2_exit, T1_exit
