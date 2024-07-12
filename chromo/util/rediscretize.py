"""Manage the re-discretization of a polymer.

Author:     Joseph Wakim
Date:       July 31, 2022
Group:      Spakowitz Lab, Stanford
"""

import os
import sys
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import chromo.polymers as poly
from chromo.util.poly_paths import gaussian_walk
from chromo.fields import UniformDensityField
import chromo.binders as bind
import chromo.mc.mc_controller as ctrl
import chromo.mc.move_funcs as mv
from chromo.mc.mc_sim import mc_step


def get_cg_bead_intervals(
    num_beads: int, cg_factor: float
) -> Dict[int, Tuple[int, int]]:
    """Get the bead intervals corresponding to each coarse-grained bead.

    Parameters
    ----------
    num_beads : int
        Number of beads in the original polymer, prior to coarse-graining
    cg_factor : float
        Factor by which to coarse-grain the original polymer

    Returns
    -------
    Dict[int, Tuple[int, int]]
        Dictionary of bead intervals associated with each bead in the coarse-
        grained polymer; all intervals include the lower bound bead index, but
        exclude the upper bound bead index, just as in numpy indexing
    """
    cg_beads = num_beads / cg_factor
    num_intervals = int(np.floor(cg_beads))
    left_over_beads = num_beads - (num_intervals * cg_factor)
    bead_intervals = {
        i: (i * cg_factor, (i+1) * cg_factor) for i in range(num_intervals)
    }
    if left_over_beads > 0:
        bead_intervals[num_intervals] = (
            bead_intervals[num_intervals-1][1],
            bead_intervals[num_intervals-1][1] + left_over_beads + 1
        )
    return bead_intervals


def get_avg_in_intervals(
    r: np.ndarray, intervals: Dict[int, Tuple[int, int]]
) -> np.ndarray:
    """Get centroid positions of polymer segments in specified intervals.

    Parameters
    ----------
    r : np.ndarray (N, 3) of float
        Positions of each bead in original polymer, prior to coarse-graining
    intervals : Dict[int, Tuple[int, int]]
        Dictionary of bead intervals associated with each bead in the coarse-
        grained polymer; all intervals include the lower bound bead index, but
        exclude the upper bound bead index, just as in numpy indexing

    Returns
    -------
    np.ndarray (M, 3) of float
        Positions of centroids for polymer segments associated with each bead
        interval in `intervals`; the number of rows equals the number of polymer
        segments included in `intervals`.
    """
    num_intervals = len(intervals)
    cg_r = np.zeros((num_intervals, 3), dtype=float)
    for ind, bounds in intervals.items():
        r_interval = r[bounds[0]:bounds[1]]
        avg_r = np.average(r_interval, axis=0)
        cg_r[ind, :] = avg_r
    return cg_r


def get_orientations_in_intervals(
    t3: np.ndarray, intervals: Dict[int, Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the average orientations in specified intervals.

    Parameters
    ----------
    t3, : np.ndarray (N, 3) of float
        t3 orientations of each bead in the original polymer, prior to coarse-
        graining
    intervals : Dict[int, Tuple[int, int]]
        Dictionary of bead intervals associated with each bead in the coarse-
        grained polymer; all intervals include the lower bound bead index, but
        exclude the upper bound bead index, just as in numpy indexing

    Returns
    -------
    t3_cg, t2_cg : np.ndarray (M, 3)
        Average t3 orientations and perpendicular t2 orientations for polymer
        segments associated with each bead interval in `intervals`; the number
        of rows equals the number of polymer segments included in `intervals`.
    """
    t3_cg = get_avg_in_intervals(t3, intervals)
    magnitudes = np.linalg.norm(t3_cg, axis=1)
    for i in range(t3_cg.shape[0]):
        t3_cg[i] /= magnitudes[i]
    t2_cg = np.zeros(t3_cg.shape)
    for i in range(t3_cg.shape[0]):
        if not np.all(np.equal(t3_cg[i], np.array([1, 0, 0]))):
            t2_cg[i] = np.cross(t3_cg[i], np.array([1, 0, 0]))
        else:
            t2_cg[i] = np.cross(t3_cg[i], np.array([0, 1, 0]))
    magnitudes = np.linalg.norm(t2_cg, axis=1)
    for i in range(t2_cg.shape[0]):
        t2_cg[i] /= magnitudes[i]
    return t3_cg, t2_cg


def get_majority_state_in_interval(
    states: np.ndarray, intervals: Dict[int, Tuple[int, int]]
) -> np.ndarray:
    """Get majority chemical modification or binding state in polymer segments.

    Parameters
    ----------
    states : np.ndarray (N, S) of int
        Chemical modification or binding states in the original polymer, before
        any coarse-graining; rows correspond to individual beads in the original
        polymer, and columns correspond to chemical modifications or binders
    intervals : Dict[int, Tuple[int, int]]
        Dictionary of bead intervals associated with each bead in the coarse-
        grained polymer; all intervals include the lower bound bead index, but
        exclude the upper bound bead index, just as in numpy indexing

    Returns
    -------
    np.ndarray (M, S) of int
        Chemical modifications or binding states of each bead in the coarse-
        grained polymer, where rows correspond to the coarse-grained beads
        (associated with the intervals of the original polymer) and columns
        corresponds to chemical modifications or binders
    """
    num_intervals = len(intervals)
    cg_states = np.zeros((num_intervals, states.shape[1]), dtype=int)
    for ind, bounds in intervals.items():
        states_interval = states[bounds[0]:bounds[1]]
        for i in range(states.shape[1]):
            if bounds[1] - bounds[0] > 1:
                cg_states[ind, i] = np.argmax(
                    np.bincount(states_interval[:, i])
                )
            else:
                cg_states[ind, i] = states_interval[0, i]
    return cg_states


def get_random_state_in_interval(
    states: np.ndarray, intervals: Dict[int, Tuple[int, int]]
) -> np.ndarray:
    """Get random chemical modification or binding state in polymer segments.

    Parameters
    ----------
    states : np.ndarray (N, S) of int
        Chemical modification or binding states in the original polymer, before
        any coarse-graining; rows correspond to individual beads in the original
        polymer, and columns correspond to chemical modifications or binders
    intervals : Dict[int, Tuple[int, int]]
        Dictionary of bead intervals associated with each bead in the coarse-
        grained polymer; all intervals include the lower bound bead index, but
        exclude the upper bound bead index, just as in numpy indexing

    Returns
    -------
    np.ndarray (M, S) of int
        Chemical modifications or binding states of each bead in the coarse-
        grained polymer, where rows correspond to the coarse-grained beads
        (associated with the intervals of the original polymer) and columns
        corresponds to chemical modifications or binders
    """
    num_intervals = len(intervals)
    cg_states = np.zeros((num_intervals, states.shape[1]), dtype=int)
    for ind, bounds in intervals.items():
        states_interval = states[bounds[0]:bounds[1]]
        for i in range(states.shape[1]):
            if bounds[1] - bounds[0] > 1:
                cg_states[ind, i] = np.random.choice(states_interval[:, i])
            else:
                cg_states[ind, i] = states_interval[0, i]
    return cg_states


def get_cg_udf(
    udf_refined_dict: Dict, binders_refined: pd.DataFrame,
    cg_factor: float, polymers_cg: List[poly.Chromatin]
) -> UniformDensityField:
    """Get Uniform Density Field corresponding to the coarse-grained polymer.

    Notes
    -----
    The coarse-grained Uniform Density Field maintains approximately the same
    number density of beads as exists in the refined field. This means that the
    simulation space and the confinement volume decreases upon coarse-graining.
    The dimensions of the voxels do not change, nor do the physical properties
    of the polymer backbone and reader proteins. For proper density
    interpolation in the field, there must be at least two voxels in each
    direction.

    Parameters
    ----------
    udf_refined_dict : Dict
        Dictionary of parameters for uniform density field corresponding to
        refined polymer; passing the actual UDF into the function is slow
    binders_refined : pd.DataFrame
        Binder collection corresponding to refined polymer
    cg_factor : float
        Degree of coarse graining; factor reduction in the number of beads
        between the refined polymer and the coarse-grained polymer
    polymers_cg : List[poly.Chromatin]
        Coarse-grained representation of chromatin fiber

    Returns
    -------
    UniformDensityField
        Uniform density field corresponding to the coarse-grained polymer
    """
    nx_cg = int(round(udf_refined_dict["nx"] / (cg_factor**(1/3))))
    ny_cg = int(round(udf_refined_dict["ny"] / (cg_factor**(1/3))))
    nz_cg = int(round(udf_refined_dict["nz"] / (cg_factor**(1/3))))

    # Verify that we maintain at least two voxels in each dimension
    if (nx_cg < 2) or (ny_cg < 2) or (nz_cg < 2):
        raise ValueError(
            f"Coarse-graining factor {cg_factor} is not supported by the field."
        )

    # If confined, add buffer layer of voxels to block periodicity
    if udf_refined_dict["confine_type"] != "":
        nx_cg += 1
        ny_cg += 1
        nz_cg += 1

    x_width_cg = nx_cg * udf_refined_dict["x_width"] / udf_refined_dict["nx"]
    y_width_cg = ny_cg * udf_refined_dict["y_width"] / udf_refined_dict["ny"]
    z_width_cg = nz_cg * udf_refined_dict["z_width"] / udf_refined_dict["nz"]

    binders_cg = get_cg_binders(binders_refined, cg_factor)

    udf_cg = UniformDensityField(
        polymers=polymers_cg,
        binders=binders_cg,
        x_width=x_width_cg,
        nx=nx_cg,
        y_width=y_width_cg,
        ny=ny_cg,
        z_width=z_width_cg,
        nz=nz_cg,
        confine_type=udf_refined_dict["confine_type"],
        confine_length=udf_refined_dict["confine_length"] / (cg_factor**(1/3)),
        chi=udf_refined_dict["chi"],
        assume_fully_accessible=udf_refined_dict["assume_fully_accessible"],
        vf_limit=udf_refined_dict["vf_limit"],
        fast_field=udf_refined_dict["fast_field"],
        n_points=udf_refined_dict["n_points"]
    )
    return udf_cg


def get_cg_udf_old(
    udf_refined_dict: Dict, binders_refined: pd.DataFrame,
    cg_factor: float, polymers_cg: List[poly.Chromatin]
) -> UniformDensityField:
    """Get Uniform Density Field corresponding to the coarse-grained polymer.

    Notes
    -----
    The coarse-grained Uniform Density Field is defined to approximately
    maintain the average volume fraction of a single bead in a single voxel.
    However, the uniform denisty field is required to have at least two voxels
    in each dimension.

    The overall simulation space is generally unaffected by coarse-graining. The
    confinement is unaffected by coarse-graining. HOWEVER, in the case that a
    confinement is active, periodic boundaries should not be engaged. To ensure
    this is the case, for confined systems, we add a layer of voxels at all
    boundaries upon coarse-graining.

    Parameters
    ----------
    udf_refined_dict : Dict
        Dictionary of parameters for uniform density field corresponding to
        refined polymer; passing the actual UDF into the function is slow
    binders_refined : pd.DataFrame
        Binder collection corresponding to refined polymer
    cg_factor : float
        Degree of coarse graining; factor reduction in the number of beads
        between the refined polymer and the coarse-grained polymer
    polymers_cg : List[poly.Chromatin]
        Coarse-grained representation of chromatin fiber

    Returns
    -------
    UniformDensityField
        Uniform density field corresponding to the coarse-grained polymer
    """
    dx = udf_refined_dict["x_width"] / udf_refined_dict["nx"]
    dy = udf_refined_dict["y_width"] / udf_refined_dict["ny"]
    dz = udf_refined_dict["z_width"] / udf_refined_dict["nz"]
    dx_cg = dx * (cg_factor**(1/3))
    dy_cg = dy * (cg_factor**(1/3))
    dz_cg = dz * (cg_factor**(1/3))
    nx_cg = int(round(udf_refined_dict["x_width"] / dx_cg))
    ny_cg = int(round(udf_refined_dict["y_width"] / dy_cg))
    nz_cg = int(round(udf_refined_dict["z_width"] / dz_cg))

    if (nx_cg < 2) or (ny_cg < 2) or (nz_cg < 2):
        raise ValueError(
            f"Coarse-graining factor {cg_factor} is not supported by the field."
        )

    # If there is no confinement, periodic boundaries are engaged
    if udf_refined_dict["confine_type"] == "":
        x_width_cg = udf_refined_dict["x_width"]
        y_width_cg = udf_refined_dict["y_width"]
        z_width_cg = udf_refined_dict["z_width"]

    # Otherwise, add a buffer layer of voxels to block periodicity
    else:
        nx_cg += 1
        ny_cg += 1
        nz_cg += 1
        x_width_cg = nx_cg * dx_cg
        y_width_cg = ny_cg * dy_cg
        z_width_cg = nz_cg * dz_cg

    binders_cg = get_cg_binders(binders_refined, cg_factor)

    udf_cg = UniformDensityField(
        polymers=polymers_cg,
        binders=binders_cg,
        x_width=x_width_cg,
        nx=nx_cg,
        y_width=y_width_cg,
        ny=ny_cg,
        z_width=z_width_cg,
        nz=nz_cg,
        confine_type=udf_refined_dict["confine_type"],
        confine_length=udf_refined_dict["confine_length"],
        chi=udf_refined_dict["chi"],
        assume_fully_accessible=udf_refined_dict["assume_fully_accessible"],
        vf_limit=udf_refined_dict["vf_limit"],
        fast_field=udf_refined_dict["fast_field"],
        n_points=udf_refined_dict["n_points"]
    )
    return udf_cg


def get_cg_binders(
    binders_refined: pd.DataFrame, cg_factor: float
) -> pd.DataFrame:
    """Get binders corresponding to the coarse-grained polymer.

    Notes
    -----
    We may want to scale the interaction volume with the size of the coarse-
    grained beads. We may also want to scale the interaction energy with the
    coarse-graining factor. For now, both parameters are kept constant, but
    we can adjust them in this function.

    Parameters
    ----------
    binders_refined : pd.DataFrame
        Binder collection corresponding to refined polymer
    cg_factor : float
        Degree of coarse graining; factor reduction in the number of beads
        between the refined polymer and the coarse-grained polymer

    Returns
    -------
    pd.DataFrame
        Table of binder properties corresponding to the coarse-grained polymer
    """
    binders_cg = binders_refined.copy()
    binders_cg["interaction_energy"] *= 1       # cg_factor**(1/6)
    binders_cg["interaction_radius"] *= 1       # cg_factor**(1/6)
    binders_cg["interaction_volume"] *= 1       # cg_factor**(1/2)
    for ind in binders_cg.index:
        binders_cg["cross_talk_interaction_energy"][ind].update(
            (x, y * 1)          # cg_factor**(1/6))
            for x, y in binders_cg["cross_talk_interaction_energy"][ind].items()
        )
    return binders_cg


def get_cg_chromatin(
    polymer: poly.Chromatin, cg_factor: float,
    name_cg: Optional[str] = "Chr_CG", random_states: Optional[bool] = False
) -> poly.Chromatin:
    """Generate a coarse-grained representation of a wormlike chain polymer.

    Notes
    -----
    The properties of monomers in the coarse-grained chain are equivalent to
    those in the original polymer. To determine the positions of the coarse-
    grained beads, first calculate the centroid position of the polymer segments
    that they represent, then scale the radial position of the bead inward by
    the degree of coarse-graining.

    Parameters
    ----------
    polymer : poly.Chromatin
        Original polymer for which a coarse-grained representation is requested
    cg_factor : float
        Degree to which the polymer will be coarse-grained; the number of beads
        in the original polymer that will be represented by a single bead in the
        coarse-grained polymer
    name_cg : Optional[str]
        Name of coarse-grained polymer (default = "Chr_CG")
    random_states : Optional[bool]
        Indicator for whether to pick modification states randomly from bead
        segments for each coarse-grained bead (True) or whether to pick the
        majority modification state from each bead segment (False; default case)

    Returns
    -------
    poly.Chromatin
        Coarse-grained representation of the original polymer, where each coarse
        grained bead is defined with the average position and orientation and
        the most common chemical modification and binder state in its segment
    """
    intervals = get_cg_bead_intervals(polymer.num_beads, cg_factor)
    r_cg = get_avg_in_intervals(polymer.r, intervals)
    t3_cg, t2_cg = get_orientations_in_intervals(polymer.t3, intervals)
    states_cg = get_majority_state_in_interval(polymer.states, intervals)
    if random_states:
        chem_mods_cg = get_random_state_in_interval(
            polymer.chemical_mods, intervals
        )
    else:
        chem_mods_cg = get_majority_state_in_interval(
            polymer.chemical_mods, intervals
        )
    # NOTE: Currently, the code accommodates a single linker length
    # TODO: Adjust the line below to account for variable linker lengths
    bead_length_cg = np.ones(len(r_cg)-1) * polymer.bead_length[0]
    bead_rad_cg = polymer.bead_rad

    # Scale the radial position of the beads inwards based on `cg_factor`
    r_cg /= (cg_factor**(1/3))

    polymer_cg = poly.Chromatin(
        name=name_cg,
        r=r_cg,
        bead_length=bead_length_cg,
        bead_rad=bead_rad_cg,
        t3=t3_cg,
        t2=t2_cg,
        states=states_cg,
        binder_names=polymer.binder_names,
        chemical_mods=chem_mods_cg,
        chemical_mod_names=polymer.chemical_mod_names,
        log_path=polymer.log_path,
        max_binders=polymer.max_binders
    )
    return polymer_cg


def get_cg_chromatin_old(
    polymer: poly.Chromatin, cg_factor: float, name_cg: Optional[str] = "Chr_CG"
) -> poly.Chromatin:
    """Generate a coarse-grained representation of a wormlike chain polymer.

    Notes
    -----
    The nominal bead length is of the coarse-grained polymer is determined from
    the expected end-to-end distance of a segment `cg_factor` in size.

    The bead radius of the coarse-grained polymer is determined to maintain the
    bead volumes upon coarse-graining.

    Parameters
    ----------
    polymer : poly.Chromatin
        Original polymer for which a coarse-grained representation is requested
    cg_factor : float
        Degree to which the polymer will be coarse-grained; the number of beads
        in the original polymer that will be represented by a single bead in the
        coarse-grained polymer
    name_cg : Optional[str]
        Name of coarse-grained polymer (default = "Chr_CG")

    Returns
    -------
    poly.SSWLC
        Coarse-grained representation of the original polymer, where each coarse
        grained bead is defined with the average position and orientation and
        the most common chemical modification and binder state in its segment
    """
    intervals = get_cg_bead_intervals(polymer.num_beads, cg_factor)
    r_cg = get_avg_in_intervals(polymer.r, intervals)
    t3_cg = get_avg_in_intervals(polymer.t3, intervals)
    t2_cg = get_avg_in_intervals(polymer.t2, intervals)
    states_cg = get_majority_state_in_interval(polymer.states, intervals)
    chem_mods_cg = get_majority_state_in_interval(
        polymer.chemical_mods, intervals
    )
    lp = polymer.lp
    bead_length_cg = np.sqrt(2 * lp**2 * (
        (cg_factor * polymer.bead_length) / lp - 1 + np.exp(
            -((cg_factor * polymer.bead_length) / lp)
        )
    ))
    bead_rad_cg = polymer.bead_rad * cg_factor**(1/3)
    polymer_cg = poly.Chromatin(
        name=name_cg,
        r=r_cg,
        bead_length=bead_length_cg,
        bead_rad=bead_rad_cg,
        t3=t3_cg,
        t2=t2_cg,
        states=states_cg,
        binder_names=polymer.binder_names,
        chemical_mods=chem_mods_cg,
        chemical_mod_names=polymer.chemical_mod_names,
        log_path=polymer.log_path,
        max_binders=polymer.max_binders
    )
    return polymer_cg


def get_refined_intervals(
    cg_r: np.ndarray, num_beads_cg: int, num_beads_refined: int
) -> Tuple[Dict[int, int], Dict[int, Tuple[float, float]]]:
    """Get the number of refined beads to assign to each coarse-grained bead.

    Parameters
    ----------
    cg_r : np.ndarray (M, 3) of float
        Coarse-grained bead positions; rows indicate beads in the coarse-grained
        polymer, and columns correspond to (x, y, z) coordinates
    num_beads_cg : int
        Number of beads in the coarse-grained polymer
    num_beads_refined : int
        Number of beads in the refined polymer

    Returns
    -------
    num_steps : Dict[int, int]
        Number of refined beads to assign to each coarse-grained bead; keys
        identify index of coarse-grained bead, values indicate number of beads
        in refined polymer assigned to the coarse-grained bead
    start_end : Dict[int, Tuple[int, int]]
        Starting and ending positions of each segment in the refined polymer
        corresponding to the beads in the coarse-grained polymer; free ends
        do not have a fixed position and are indicated with `np.nan` positions
    """
    seg_size = int(np.floor(num_beads_refined / (num_beads_cg-1)))
    seg_half_1 = int(np.floor(seg_size/2))
    seg_half_2 = seg_size - seg_half_1
    left_over = num_beads_refined % (num_beads_cg-1)

    num_steps = {0: seg_half_1}
    for i in range(1, num_beads_cg-1):
        num_steps[i] = seg_size
    num_steps[num_beads_cg-1] = seg_half_2 - 1

    start_end = {0: (np.asarray([np.nan, np.nan, np.nan]), cg_r[0, :])}
    for i in range(1, num_beads_cg):
        start_end[i] = (cg_r[i-1, :], cg_r[i, :])

    if left_over > 0:
        num_steps[num_beads_cg] = left_over
        start_end[num_beads_cg] = (
            cg_r[num_beads_cg-1, :], np.asarray([np.nan, np.nan, np.nan])
        )

    return num_steps, start_end


def brownian_bridge(
    N: int, p0: np.ndarray, p1: np.ndarray,
    avg_step_target: Optional[float] = None
) -> np.ndarray:
    """Create a Brownian Bridge with N steps.

    Adapted from https://gist.github.com/delta2323/6bb572d9473f3b523e6e, which
    provides code for a Brownian Bridge. This function can be used for
    orientations as well.

    Notes
    -----
    Let X, Y, and Z be successive points on the path directly connecting p0 and
    p1. Let B[0,:], B[1,:], and B[2,:] be deviations to the X, Y, and Z points,
    respectively, along that path. We want to adjust B[0,:], B[1,:], and B[2,:]
    so that our average step from p0 to p1 is of size `avg_step_target`.
    Therefore, for N steps:

    sum_{i=1}^{N} sqrt(
        (B[0,i] + X[i])^2 + (B[1,i] + Y[i])^2 + (B[2,i] + Z[i])^2
    ) / N = avg_step_target

    avg_step_target * N = sum_{i=1}^{N} sqrt(
        (B[0,i] + X[i])^2 + (B[1,i] + Y[i])^2 + (B[2,i] + Z[i])^2
    )

    avg_step_target * N = sum_{i=1}^{N} sqrt(
        B[0,i]^2 + 2B[0,i]X[i] + X[i]^2 +
        B[1,i]^2 + 2B[1,i]Y[i] + Y[i]^2 +
        B[2,i]^2 + 2B[2,i]Z[i]+ Z[i]^2
    )

    We approximate the solution to this by assuming that B[0,i] >> X[i],
    B[1,i] >> Y[i], and B[2,i] >> Z[i]. Therefore:

    avg_step_target * N \approx sum_{i=1}^{N} sqrt(
        B[0,i]^2 + B[1,i]^2 + B[2,i]
    )

    The average magnitude of each row in B needs to be `avg_step_target`.

    Parameters
    ----------
    N : int
        Number of steps in path from `p0` to `p1`
    p0, p1 : np.ndarray (3,) of float
        Original and final positions in segment path, respectively
    avg_step_target : Optional[float]
        Approximate average step size to make on path between points `p0` and
        `p1` (default = None; indicating not to rescale steps of the Brownian
        bridge)

    Returns
    -------
    np.ndarray (N, 3) of float
        Path of N points connecting points `p0` and `p1`
    """
    # Trivial case: single step
    if N == 1:
        return np.array([p0, p1])
    # If not trivial case, extend N by 1, since last index is not included
    N += 1
    # Generate Brownian Bridge
    dt = 1.0 / (N - 1)
    dt_sqrt = np.sqrt(dt)
    B = np.empty((N, 3), dtype=float)
    B[0] = 0
    for n in range(N-2):
        t = n * dt
        xi = np.random.randn(3) * dt_sqrt
        B[n+1, :] = B[n, :] * (1 - dt / (1 - t)) + xi
    B[N-1, :] = np.array([0, 0, 0])
    # Characterize the direct path connecting the two points
    direct_path_length = np.linalg.norm(p1 - p0)
    direct_step = direct_path_length / (N - 1)
    direct_path = np.linspace(p0, p1, N)
    # Scale `B` so it is >> average steps on direct path
    if avg_step_target is not None:
        B *= direct_path_length
    # Generate an actual path connecting the two points
    path = direct_path + B
    actual_step_sizes = np.linalg.norm(
        np.diff(path, axis=0), axis=1
    )
    total_path_length = np.sum(actual_step_sizes)
    avg_step_size = total_path_length / (N - 1)
    # Normalize avg step size along path
    if avg_step_target is not None:
        if avg_step_target < direct_step:
            print(
                f"Points cannot be connected with steps of {avg_step_target}."
            )
            print(f"The minimum step size required is {direct_step}.")
            print(f"The bead spacing will be adjusted to {direct_step}.")
            avg_step_size = direct_step
        actual_to_direct = avg_step_size / direct_step
        target_to_direct = avg_step_target / direct_step
        # Approximate steps mostly in direction of `B`
        path = direct_path + (B / (actual_to_direct / target_to_direct))
    return path


def gaussian_walk_from_point(start, N, step_size):
    """Define a Gaussian random walk from specified starting point

    Parameters
    ----------
    start : np.ndarray (3,) of float
        Starting point of Gaussian random walk
    N : int
        Number of points in the Gaussian random walk
    step_size : float
        Step size in the Gaussian random walk

    Returns
    -------
    np.ndarray (3, N) of float
        Gaussian random walk from specified starting point
    """
    return gaussian_walk(N, step_size) + start


def enforce_spherical_confinement(r: np.ndarray, rad: float) -> np.ndarray:
    """Enforce a spherical confinement when generating initial refined chains

    Parameters
    ----------
    r : np.ndarray (N, 3) of float
        Positions of beads in refined chain
    rad : float
        Radius of the spherical confinement

    Returns
    -------
    np.ndarray (N, 3) of float
        Array of refined bead positions, adjusted to satisfy the spherical
        confinement
    """
    kernel = np.array([0.98, 0.97, 0.96, 0.95, 0.96, 0.97, 0.98])
    violators = []
    factor_violation = []
    num_r = len(r)
    num_r_m4 = num_r - 4
    num_kernel = len(kernel)
    for i in range(num_r):
        dist = np.linalg.norm(r[i])
        if dist > rad:
            violators.append(i)
            factor_violation.append(dist/rad)
    violators = np.array(violators)
    factor_violation = np.array(factor_violation)
    for i, ind in enumerate(violators):
        if (ind >= 3) and (ind <= num_r_m4):
            for j in range(3):
                r[ind-3:ind+4, j] = np.multiply(
                    r[ind-3:ind+4, j], kernel * (1/factor_violation[i])
                )
        elif ind < 3:
            for j in range(3):
                r[:ind+4, j] = np.multiply(
                    r[:ind+4, j], kernel[3-ind:] * (1/factor_violation[i])
                )
        elif ind > num_r_m4:
            for j in range(3):
                r[ind-3:, j] = np.multiply(
                    r[ind-3:,j], kernel[:num_r_m4-ind] * (1/factor_violation[i])
                )
    return r


def get_refined_path(
    cg_r: np.ndarray, num_beads_refined: int,
    bead_spacing: Optional[float] = np.pi
) -> np.ndarray:
    """Get the path of a refined polymer.

    Notes
    -----
    This function can be used for orientations as well.

    Parameters
    ----------
    cg_r : np.ndarray (M, 3) of float
        Coarse-grained bead positions; rows indicate beads in the coarse-grained
        polymer, and columns correspond to (x, y, z) coordinates
    num_beads_refined : int
        Number of beads in the refined chromatin fiber
    bead_spacing : Optional[float]
        Average bead spacing in refined polymer configuration (default =
        np.pi; relevant when refining orientation vectors)

    Returns
    -------
    np.ndarray (N, 3) of float
        Path of the refined polymer, where rows correspond to beads and columns
        correspond to (x, y, z) positions.
    """
    num_beads_cg = len(cg_r)
    num_steps, start_end = get_refined_intervals(
        cg_r, num_beads_cg, num_beads_refined
    )
    refined_path = []
    for ind, bounds in start_end.items():
        # First and last segments are free ends
        if ind == 0:
            seg_path = np.flip(gaussian_walk_from_point(
                bounds[1], num_steps[ind],
                np.array([bead_spacing] * num_steps[ind])
            ), axis=0)[:num_steps[ind]]
        elif ind == num_beads_cg:
            seg_path = gaussian_walk_from_point(
                bounds[0], num_steps[ind],
                np.array([bead_spacing] * num_steps[ind])
            )
        # Intermediate segments have constrained points
        else:
            seg_path = brownian_bridge(
                num_steps[ind], np.asarray(bounds[0]), np.asarray(bounds[1]),
                avg_step_target=bead_spacing
            )[:num_steps[ind]]
        refined_path.append(seg_path)
    return np.vstack(refined_path)


def get_refined_orientations(
    t3_cg: np.ndarray, num_beads_refined: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the orientation vectors for the refined polymer

    Parameters
    ----------
    t3_cg : np.ndarray (M, 3) of float
        Coarse-grained bead orientations; rows indicate beads in the coarse-
        grained polymer, and columns correspond to (x, y, z) t3 orientations
    num_beads_refined : int
        Number of beads in the refined chromatin fiber

    Returns
    -------
    t3_refined, t2_refined : np.ndarray (N, 3) of float
        t3 of the refined polymer, where rows correspond to beads and columns
        correspond to (x, y, z) components of the orientations.
    """
    t3_refined = get_refined_path(t3_cg, num_beads_refined)
    magnitudes = np.linalg.norm(t3_refined, axis=1)
    for i in range(t3_refined.shape[0]):
        t3_refined[i] /= magnitudes[i]
    t2_refined = np.zeros(t3_refined.shape)
    for i in range(t3_refined.shape[0]):
        if not np.all(np.equal(t3_refined[i], np.array([1, 0, 0]))):
            t2_refined[i] = np.cross(t3_refined[i], np.array([1, 0, 0]))
        else:
            t2_refined[i] = np.cross(t3_refined[i], np.array([0, 1, 0]))
    magnitudes = np.linalg.norm(t2_refined, axis=1)
    for i in range(t2_refined.shape[0]):
        t2_refined[i] /= magnitudes[i]
    return t3_refined, t2_refined


def refine_udf(
    udf_cg: UniformDensityField, binders_cg: pd.DataFrame, cg_factor: float,
    polymers_refined: List[poly.Chromatin]
) -> UniformDensityField:
    """Generate the Uniform Density Field for the refined polymer.

    Parameters
    ----------
    udf_cg : UniformDensityField
        Uniform density field corresponding to the coarse-grained polymer
    binders_cg : pd.DataFrame
        Binder collection corresponding to refined polymer
    cg_factor : float
        Degree of coarse graining; factor reduction in the number of beads
        between the refined polymer and the coarse-grained polymer
    polymers_refined : List[poly.Chromatin]
        Refined representation of chromatin fiber

    Returns
    -------
    UniformDensityField
        Uniform density field corresponding to the refined polymer
    """
    nx_refined = int(round(udf_cg.nx * (cg_factor**(1/3))))
    ny_refined = int(round(udf_cg.ny * (cg_factor**(1/3))))
    nz_refined = int(round(udf_cg.nz * (cg_factor**(1/3))))

    # If confined, add buffer layer of voxels to block periodicity
    if udf_cg.confine_type != "":
        nx_refined += 1
        ny_refined += 1
        nz_refined += 1

    x_width_refined = nx_refined * udf_cg.dx
    y_width_refined = ny_refined * udf_cg.dy
    z_width_refined = nz_refined * udf_cg.dz

    binders_refined = refine_binders(binders_cg, cg_factor)
    udf_refined = UniformDensityField(
        polymers=polymers_refined,
        binders=binders_refined,
        x_width=x_width_refined,
        nx=nx_refined,
        y_width=y_width_refined,
        ny=ny_refined,
        z_width=z_width_refined,
        nz=nz_refined,
        confine_type=udf_cg.confine_type,
        confine_length=udf_cg.confine_length * (cg_factor**(1/3)),
        chi=udf_cg.chi,
        assume_fully_accessible=udf_cg.assume_fully_accessible,
        vf_limit=udf_cg.vf_limit,
        fast_field=udf_cg.fast_field,
        n_points=udf_cg.n_points
    )
    return udf_refined


def refine_udf_old(
    udf_cg: UniformDensityField, binders_cg: pd.DataFrame, cg_factor: float,
    polymers_refined: List[poly.Chromatin]
) -> UniformDensityField:
    """Generate the Uniform Density Field for the refined polymer.

    Parameters
    ----------
    udf_cg : UniformDensityField
        Uniform density field corresponding to the coarse-grained polymer
    binders_cg : pd.DataFrame
        Binder collection corresponding to refined polymer
    cg_factor : float
        Degree of coarse graining; factor reduction in the number of beads
        between the refined polymer and the coarse-grained polymer
    polymers_refined : List[poly.Chromatin]
        Refined representation of chromatin fiber

    Returns
    -------
    UniformDensityField
        Uniform density field corresponding to the refined polymer
    """
    dx_refined = udf_cg.dx / (cg_factor**(1/3))
    dy_refined = udf_cg.dy / (cg_factor**(1/3))
    dz_refined = udf_cg.dz / (cg_factor**(1/3))
    nx_refined = int(round(udf_cg.x_width / dx_refined))
    ny_refined = int(round(udf_cg.y_width / dy_refined))
    nz_refined = int(round(udf_cg.z_width / dz_refined))

    x_width_refined = dx_refined * nx_refined
    y_width_refined = dy_refined * ny_refined
    z_width_refined = dz_refined * nz_refined

    binders_refined = refine_binders(binders_cg, cg_factor)
    udf_refined = UniformDensityField(
        polymers=polymers_refined,
        binders=binders_refined,
        x_width=x_width_refined,
        nx=nx_refined,
        y_width=y_width_refined,
        ny=ny_refined,
        z_width=z_width_refined,
        nz=nz_refined,
        confine_type=udf_cg.confine_type,
        confine_length=udf_cg.confine_length,
        chi=udf_cg.chi,
        assume_fully_accessible=udf_cg.assume_fully_accessible,
        vf_limit=udf_cg.vf_limit,
        fast_field=udf_cg.fast_field,
        n_points=udf_cg.n_points
    )
    return udf_refined


def refine_binders(binders_cg: pd.DataFrame, cg_factor: float) -> pd.DataFrame:
    """Generate binders for the refined polymer.

    Notes
    -----
    We may want to scale the interaction volume with the degree of refinement.
    We may also want to scale the interaction energy with the degree of
    refinement. For now, both parameters are kept constant, but we can adjust
    them in this function.

    Parameters
    ----------
    binders_cg : pd.DataFrame
        Binder collection corresponding to refined polymer
    cg_factor : float
        Degree of coarse graining; factor reduction in the number of beads
        between the refined polymer and the coarse-grained polymer

    Returns
    -------
    pd.DataFrame
        Table of binder properties corresponding to the refined polymer
    """
    binders_refine = binders_cg.copy()
    binders_refine["interaction_energy"] /= 1       # cg_factor**(1/6)
    binders_refine["interaction_radius"] /= 1       # cg_factor**(1/6)
    binders_refine["interaction_volume"] /= 1       # cg_factor**(1/2)
    for ind in binders_refine.index:
        binders_refine["cross_talk_interaction_energy"][ind].update(
            (x, y / 1)      # cg_factor**(1/6))
            for x, y in
            binders_refine["cross_talk_interaction_energy"][ind].items()
        )
    return binders_refine


def refine_chromatin(
    polymer_cg: poly.Chromatin, num_beads_refined: int, bead_spacing: float,
    chemical_mods: np.ndarray, udf_cg: UniformDensityField,
    binding_equilibration: Optional[int] = 0,
    name_refine: Optional[str] = "Chr", output_dir: Optional[str] = "."
) -> Tuple[poly.Chromatin, UniformDensityField]:
    """Refined chromatin configuration from coarse-grained representation.

    Notes
    -----
    Beads are extended from the ends of the coarse-grained representation using
    a Gaussian random walk.

    Beads in the coarse-grained representation are connected using Brownian
    bridges.

    Binding states of beads in the refined polymer are initialized as unbound,
    then provided `binding_equilibration` moves to re-equilibrate. Binding
    re-equilibration is performed before any MC moves are attempted on the
    refined polymer.

    Chemical modification states for the refined polymer must be specified.

    The physical properties of all beads are maintained upon refinement.

    Parameters
    ----------
    polymer_cg : poly.Chromatin
        Coarse-grained representation of the chromatin fiber
    num_beads_refined : int
        Number of beads in the refined chromatin fiber
    bead_spacing: float
        Average spacing of final polymer
    chemical_mods : np.ndarray (N, M) of int
        Chemical modification states of the refined polymer, where rows
        correspond to individual beads in the refined polymer and columns
        correspond to the number of chemical modification types on each bead
    udf_cg: UniformDensityField
        Uniform density field corresponding to the coarse-grained polymer
    binding_equilibration : Optional[int]
        Number of binding state moves to equilibrate reader protein binding on
        the refined chromatin fiber; should be >= to 0 (Default = 0, indicating
        that no binding moves are attempted on the refined polymer)
    name_refine : Optional[str]
        Name of the refined polymer (default = "Chr")
    output_dir : Optional[str]
        Output directory at which to save refined polymer configuration (default
        = ".")

    Returns
    -------
    polymer_refined : polymer.Chromatin
        Refined chromatin representation derived from the coarse-grained polymer
    UniformDensityField : UniformDensityField
        Uniform density field corresponding to the refined polymer
    """
    num_beads_cg = len(polymer_cg.r)
    scaling = (num_beads_refined / num_beads_cg)**(1/3)
    bead_spacing_scaled_inward = \
        np.ones(num_beads_cg-1) * bead_spacing / scaling
    r_refine = get_refined_path(
        polymer_cg.r, num_beads_refined, bead_spacing_scaled_inward[0]
    )
    t3_refine, t2_refine = get_refined_orientations(
        polymer_cg.t3, num_beads_refined
    )
    states_refine = np.zeros(
        (num_beads_refined, chemical_mods.shape[1]), dtype=int
    )

    # Scale the radial position of the beads outward based on `cg_factor`
    r_refine *= scaling

    # NOTE: Currently, the code accommodates a single linker length
    # TODO: Adjust the line below to account for variable linker lengths
    bead_spacing_refine = np.ones(len(r_refine)-1) * bead_spacing

    polymer = poly.Chromatin(
        name=name_refine,
        r=r_refine,
        bead_length=bead_spacing_refine,
        bead_rad=polymer_cg.bead_rad,
        t3=t3_refine,
        t2=t2_refine,
        states=states_refine,
        binder_names=polymer_cg.binder_names,
        chemical_mods=chemical_mods,
        chemical_mod_names=polymer_cg.chemical_mod_names,
        max_binders=polymer_cg.max_binders
    )
    udf = refine_udf(
        udf_cg, udf_cg.binders, num_beads_refined/num_beads_cg, [polymer]
    )

    # Enforce spherical confinement, if specified
    if udf.confine_type == "Spherical":
        polymer.r = enforce_spherical_confinement(
            polymer.r, udf.confine_length
        )

    if binding_equilibration > 0:
        binding_move = ctrl.specific_move(
            mv.change_binding_state,
            log_dir=output_dir,
            bead_amp_bounds={"change_binding_state": (1, 1)},
            move_amp_bounds={"change_binding_state": (1, 1)},
            controller=ctrl.NoControl
        )
        for _ in range(binding_equilibration):
            mc_step(
                binding_move[0].move, polymer, udf.binders, udf, active_field=1,
                update_distances=0
            )
    return polymer, udf


def refine_chromatin_old(
    polymer_cg: poly.Chromatin, num_beads_refined: int, bead_spacing: float,
    chemical_mods: np.ndarray, udf_cg: UniformDensityField,
    binding_equilibration: Optional[int] = 0,
    name_refine: Optional[str] = "Chr", output_dir: Optional[str] = "."
) -> Tuple[poly.Chromatin, UniformDensityField]:
    """Refined chromatin configuration from coarse-grained representation.

    Notes
    -----
    Beads are extended from the ends of the coarse-grained representation using
    a Gaussian random walk.

    Beads in the coarse-grained representation are connected using Brownian
    bridges.

    Binding states of beads in the refined polymer are initialized as unbound,
    then provided `binding_equilibration` moves to re-equilibrate. Binding
    re-equilibration is performed before any MC moves are attempted on the
    refined polymer.

    Chemical modification states for the refined polymer must be specified.

    Bead radius is selected to maintain the overall volume of beads.

    Parameters
    ----------
    polymer_cg : poly.Chromatin
        Coarse-grained representation of the chromatin fiber
    num_beads_refined : int
        Number of beads in the refined chromatin fiber
    bead_spacing: float
        Average spacing of final polymer
    chemical_mods : np.ndarray (N, M) of int
        Chemical modification states of the refined polymer, where rows
        correspond to individual beads in the refined polymer and columns
        correspond to the number of chemical modification types on each bead
    udf_cg: UniformDensityField
        Uniform density field corresponding to the coarse-grained polymer
    binding_equilibration : Optional[int]
        Number of binding state moves to equilibrate reader protein binding on
        the refined chromatin fiber; should be >= to 0 (Default = 0, indicating
        that no binding moves are attempted on the refined polymer)
    name_refine : Optional[str]
        Name of the refined polymer (default = "Chr")
    output_dir : Optional[str]
        Output directory at which to save refined polymer configuration (default
        = ".")

    Returns
    -------
    polymer_refined : polymer.Chromatin
        Refined chromatin representation derived from the coarse-grained polymer
    UniformDensityField : UniformDensityField
        Uniform density field corresponding to the refined polymer
    """
    num_beads_cg = len(polymer_cg.r)
    r_refine = get_refined_path(polymer_cg.r, num_beads_refined, bead_spacing)
    t3_refine = get_refined_path(polymer_cg.t3, num_beads_refined)
    t2_refine = get_refined_path(polymer_cg.t2, num_beads_refined)
    states_refine = np.zeros(
        (num_beads_refined, chemical_mods.shape[1]), dtype=int
    )
    polymer = poly.Chromatin(
        name=name_refine,
        r=r_refine,
        bead_length=bead_spacing,
        bead_rad=polymer_cg.bead_rad * (num_beads_cg/num_beads_refined)**(1/3),
        t3=t3_refine,
        t2=t2_refine,
        states=states_refine,
        binder_names=polymer_cg.binder_names,
        chemical_mods=chemical_mods,
        chemical_mod_names=polymer_cg.chemical_mod_names,
        max_binders=polymer_cg.max_binders
    )
    udf = refine_udf_old(
        udf_cg, udf_cg.binders, num_beads_refined/num_beads_cg, [polymer]
    )
    if binding_equilibration > 0:
        binding_move = ctrl.specific_move(
            mv.change_binding_state,
            log_dir=output_dir,
            bead_amp_bounds={"change_binding_state": (1, 1)},
            move_amp_bounds={"change_binding_state": (1, 1)},
            controller=ctrl.NoControl
        )
        for _ in range(binding_equilibration):
            mc_step(
                binding_move[0].move, polymer, udf.binders, udf, active_field=1
            )
    return polymer, udf
