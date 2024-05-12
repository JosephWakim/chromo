# cython: profile=True

"""Functions for performing Monte Carlo transformations.

Notes
-----
TODO: No need to make a numpy inds array when the inds are continuous!
"""

import pyximport
pyximport.install()

# Built-in Modules
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt
import sys

# External Modules
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8

# Custom Modules
import chromo.util.bead_selection as beads
cimport chromo.util.bead_selection as beads
import chromo.util.linalg as la
cimport chromo.util.linalg as la
import chromo.polymers as poly
cimport chromo.polymers as poly
from chromo.binders import Binder
from chromo.binders cimport Binder


cdef double[:] origin = np.array([0., 0., 0.])


# Crank Shaft Move

cpdef long[:] crank_shaft(
    poly.PolymerBase polymer, double amp_move, long amp_bead
):
    """Rotate section of polymer around axis connecting two end beads.
    
    Notes
    -----
    The crank shaft move affects a continuous range of beads; the internal
    configuration is unaffected by the crank shaft move. Therefore, when
    evaluating polymer energy change associated with the move, it is sufficient
    to look only at the ends of the affected segment.

    Begin by randomly selecting a starting and ending index for the crank-shaft
    move based on the bead amplitude. Then generate a random rotation angle for
    the move based on the move amplitude. Obtain the axis of rotation from the
    change in position between the starting and ending beads. Finally, generate
    the rotation matrix and obtain trial positions and tangents for evaluation.
    
    TODO: If greater speed boost is needed, return the length of `inds`.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object
    amp_move : double
        Maximum amplitude (rotation angle) of the crank-shaft move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the crank-shaft move

    Returns
    -------
    array_like (M,) of long
        Indices of M beads affected by the MC move
    """
    # print("Crank-shaft Move")
    # Type Declarations
    cdef double ang, ele_r, ele_t3, ele_t2
    cdef long bound_0, bound_1, ind0, indf, n_inds, i, j, k
    cdef long[:] inds

    # Stochastic move selection
    ang = pick_random_amp(amp_move)
    bound_0 = <int>((<double>rand() / RAND_MAX) * polymer.num_beads)
    bound_1 = beads.from_point(amp_bead, polymer.num_beads, bound_0)
    bound_1 = max(bound_1, 1)
    ind0, indf = beads.check_bead_bounds(bound_0, bound_1, polymer.num_beads)
    inds = np.arange(ind0, indf)
    n_inds = indf - ind0
    polymer.last_amp_bead = n_inds
    polymer.last_amp_move = ang

    # Prep for Deterministic Transformation
    get_crank_shaft_axis(polymer, ind0, indf)
    get_crank_shaft_fulcrum(polymer, ind0, indf)
    # noinspection PyTypeChecker
    la.arbitrary_axis_rotation(
        polymer.direction, polymer.point, ang, polymer
    )

    # "Efficient" matrix multiplication
    transform_r_t3_t2(polymer, inds, n_inds)

    return inds


cdef double pick_random_amp(double amp_move):
    """Select a random positive or negative amplitude for an MC move.
    
    Parameters
    ----------
    amp_move : double
        Range of maximum amplitude for the move; in practice, the amplitude 
        of the move is selected from between +/- amp_move/2

    Returns
    -------
    double
        Random amplitude selected for the move
    """
    return amp_move * ((<double>rand() / RAND_MAX) - 0.5)


cpdef void transform_r_t3_t2(
        poly.PolymerBase polymer,
        long[:] inds,
        long n_inds
):
    """Apply transformation operations to `r_trial`, `t3_trial`, `t2_trial`.
    
    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer affected by the transformation (which is also storing the
        transformation matrix).
    inds : long[:]
        Bead indices affected by the MC move
    n_inds : long
        Number of indices affected by the move
    """
    cdef double ele_r, ele_t3, ele_t2
    for i in range(n_inds):
        for j in range(3):
            ele_r = 0
            ele_t3 = 0
            ele_t2 = 0
            for k in range(3):
                ele_r += polymer.transformation_mat[j, k] *\
                    polymer.r[inds[i], k]
                ele_t3 += polymer.transformation_mat[j, k] *\
                    polymer.t3[inds[i], k]
                ele_t2 += polymer.transformation_mat[j, k] *\
                    polymer.t2[inds[i], k]
            polymer.r_trial[inds[i], j] =\
                ele_r + polymer.transformation_mat[j, 3]
            polymer.t3_trial[inds[i], j] = ele_t3
            polymer.t2_trial[inds[i], j] = ele_t2


cdef void get_crank_shaft_axis(
    poly.PolymerBase polymer, long ind0, long indf
):
    """Get the axis of rotation for the crank shaft move from bead selection.
    
    Notes
    -----
    For-loops are less elegant, but quicker than element-wise numpy operations
    once compiled. Our goal is to avoid python objects as much as possible. To
    do so, we are using memoryviews, which do not support element-wise 
    operations.
    
    The following cases affect the axis of rotation for the crank-shaft move:

        - CASE 1: Only first bead selected -> axis defined by first two beads
        - CASE 2: Only last bead selected -> axis defined by last two beads
        - CASE 3: Full chain selected -> axis defined by first and last beads
        - CASE 4: First bead selected -> axis defined by first bead and first 
            non-selected neighboring bead
        - CASE 5: Last bead selected -> axis defined by last non-selected 
            neighboring bead and last bead
        - CASE 6: Internal beads selected -> axis defined by last non-selected
            neighboring bead on left & first non-selected neighboring bead on
            right

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object for which crank-shaft move is applied
    ind0 : long
        Starting bead index for crank-shaft move
    indf : long
        One past ending bead index for crank-shaft move
    """
    cdef long i, indf_m1, ind0_m1, n_beads_m1, ind0_p2, ind0_m2
    cdef double scaling, magnitude

    indf_m1 = indf - 1
    ind0_m1 = ind0 - 1
    ind0_m2 = ind0 - 2
    ind0_p2 = ind0 + 2
    n_beads_m1 = polymer.num_beads - 1

    # CASE 1: Only first bead selected
    if ind0 == indf_m1 and ind0 == 0:
        for i in range(3):
            polymer.direction[i] = polymer.r[indf, i] - polymer.r[ind0, i]
    # CASE 2: Only last bead selected
    elif ind0 == indf_m1 and ind0 == n_beads_m1:
        for i in range(3):
            polymer.direction[i] = polymer.r[ind0, i] - polymer.r[ind0_m1, i]
    # CASE 3: Full chain selected
    elif ind0 == 0 and indf == polymer.num_beads:
        for i in range(3):
            polymer.direction[i] = polymer.r[indf_m1, i] - polymer.r[ind0, i]
    # CASE 4: First bead selected
    elif ind0 == 0:
        for i in range(3):
            polymer.direction[i] = polymer.r[indf, i] - polymer.r[ind0, i]
    # CASE 5: Last bead selected
    elif indf == polymer.num_beads:
        for i in range(3):
            polymer.direction[i] = polymer.r[indf_m1,i] - polymer.r[ind0_m1,i]
    # CASE 6: Internal beads selected
    else:
        for i in range(3):
            polymer.direction[i] = polymer.r[indf, i] - polymer.r[ind0_m1, i]

    magnitude = sqrt(
        polymer.direction[0]**2 + polymer.direction[1]**2 +
        polymer.direction[2]**2
    )
    if magnitude < 1E-5:
        polymer.direction = la.uniform_sample_unit_sphere()
    else:
        scaling = 1.0 / magnitude
        for i in range(3):
            polymer.direction[i] = polymer.direction[i] * scaling


cdef void get_crank_shaft_fulcrum(
    poly.PolymerBase polymer, long ind0, long indf
):
    """Get the folcrum about which to rotate beads for the crank shaft move.

    Notes
    -----
    Selection of the fulcrum for the crank shaft move is affected by which 
    beads are selected for the move. Consider the following cases:

        - CASE 1: `ind0` is 0 and full chain not selected -> folcrum is `indf`
        - CASE 2: `indf` is `polymer.num_beads` and full chain not selected ->
            folcrum is `ind0` - 1
        - CASE 3: `ind0` is 0 and `indf` is `polymer.num_beads` -> fulcrum is 
            `ind0`
        - CASE 4: Internal beads selected -> fulcrum is `ind0` - 1

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object affected by the crank-shaft move
    ind0 : long
        Starting bead index for crank-shaft move
    indf : long
        One past ending bead index for crank-shaft move
    """
    cdef long ind0_m1 = ind0 - 1

    # CASE 1: First bead and not full chain selected
    if ind0 == 0 and indf != polymer.num_beads:
        for i in range(3):
            polymer.point[i] = polymer.r[indf, i]
    # CASE 2: Last bead and not full chain selected
    elif ind0 != 0 and indf == polymer.num_beads:
        for i in range(3):
            polymer.point[i] = polymer.r[ind0_m1, i]
    # CASE 3: Full chain selected
    elif ind0 == 0 and indf == polymer.num_beads:
        for i in range(3):
            polymer.point[i] = polymer.r[ind0, i]
    # CASE 4: Internal beads selected
    else:
        for i in range(3):
            polymer.point[i] = polymer.r[ind0_m1, i]


# End Pivot Move

cpdef long[:] end_pivot(
    poly.PolymerBase polymer, double amp_move, long amp_bead
):
    """Randomly rotate segment from end of polymer about random axis.

    Notes
    -----
    Begin by selecting a random rotation angle based on the move amplitude. 
    The isolate a random subset of beads on either the LHS or RHS of the 
    polymer; the size of this subset is affected by the bead amplitude. 
    Update the transformation matrix to reflect the end-pivot of this polymer
    segment about a random rotation axis. Apply the transformation matrix to 
    the polymer position and orientations to generate trial positions and 
    orientations dictated by the move.
    
    TODO: Remove the modulo operation when determining whether or not to 
    rotate the LHS of the polymer. Set `rotate_lhs = rand()`. Change the 
    `rotate_lhs` condition to `if rotate_lhs < RAND_MAX/2`.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object
    amp_move : double
        Maximum amplitude (rotation angle) of the end-pivot move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the end-pivot move

    Returns
    -------
    array_like (M,) of long
        Indices of M beads affected by the MC move
    """
    # Type Declarations
    cdef double ang, ele_r, ele_t3, ele_t2
    cdef long ind0, indf, n_inds, rotate_lhs, i, j
    cdef long[:] inds
    cdef double[:] axis, folcrum

    # Stochastic Move Selection
    ang = pick_random_amp(amp_move)
    rotate_lhs = rand() % 2
    if rotate_lhs == 1:
        ind0 = 0
        indf = beads.from_left(amp_bead, polymer.num_beads) + 1
    else:
        ind0 = beads.from_right(amp_bead, polymer.num_beads)
        indf = polymer.num_beads
    inds = np.arange(ind0, indf)
    n_inds = indf - ind0
    axis = la.uniform_sample_unit_sphere()
    polymer.last_amp_bead = n_inds
    polymer.last_amp_move = ang
    
    # Prep for Deterministic Transformation
    fulcrum = get_end_pivot_fulcrum(polymer, ind0, indf, rotate_lhs)
    la.arbitrary_axis_rotation(axis, fulcrum, ang, polymer)

    # "Efficient" matrix multiplication
    transform_r_t3_t2(polymer, inds, n_inds)

    return inds


cdef double[:] get_end_pivot_fulcrum(
    poly.PolymerBase polymer, long ind0, long indf, long rotate_lhs
):
    """Get the folcrum about which to pivot end of polymer.
    
    Notes
    -----
    The following four cases affect the selection of the fulcrum for the 
    end-pivot move:

        - CASE 1: `ind0` is 0 and full chain not selected -> folcrum is `indf`
        - CASE 2: `indf` is `polymer.num_beads` and full chain not selected ->
            folcrum is `ind0` - 1
        - CASE 3: `ind0` is 0 and `indf` is `polymer.num_beads`, rotate LHS ->
            fulcrum is `indf` - 1
        - CASE 4: `ind0` is 0 and `indf` is `polymer.num_beads`, rotate RHS ->
            fulcrum is `ind0`

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object
    ind0 : long
        Starting bead index for crank-shaft move
    indf : long
        One past ending bead index for crank-shaft move
    rotate_lhs : long
        Indicate for whether to rotate LHS (1) or RHS (0)

    Returns
    -------
    array_like (3,) of double
        Normalized vector giving orientation of rotation axis
    """
    cdef long ind0_m1, indf_m1
    
    ind0_m1 = ind0 - 1
    indf_m1 = indf - 1

    # CASE 1: First bead and not full chain selected
    if ind0 == 0 and indf != polymer.num_beads:
        return polymer.r[indf].copy()
    # CASE 2: Last bead and not full chain selected
    if ind0 != 0 and indf == polymer.num_beads:
        return polymer.r[ind0_m1].copy()
    # CASE 3: Full chain selected, rotate LHS
    if ind0 == 0 and indf == polymer.num_beads and rotate_lhs == 1:
        return polymer.r[indf_m1].copy()
    # CASE 4: Full chain selected, rotate RHS
    return polymer.r[ind0].copy()


# Slide Move

cpdef long[:] slide(poly.PolymerBase polymer, double amp_move, long amp_bead):
    """Randomly translate a segment of the polymer.
    
    Notes
    -----
    Randomly set the translation distance based on the slide move amplitude.
    Then randomly pick a direction by sampling uniformally from a unit sphere.
    Split the translation distance into x, y, z components using the direction.
    Select a random segment of beads to move based on the bead amplititude,
    and identify the affected indices. Update and apply the transformation 
    matrix for the translation to the affected beads.

    There is a difference in how we define the move amplitude between this
    code and the original FORTRAN codebase. In the original codebase, the move
    amplitude specifies the maximum translation in each dimension, while in this
    code, the move amplitude specifies the maximum magnitude of translation.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object
    amp_move : double
        Maximum amplitude (translation distance) of the slide move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the slide move

    Returns
    -------
    array_like (M,) of long
        Indices of M beads affected by the MC move
    """
    # Type Declarations
    cdef double amp
    cdef long bound_0, bound_1, ind0, indf, n_inds, i, j
    cdef long[:] inds
    cdef double[:] direction, delta

    # Stochastic Move Selection
    amp = amp_move * (<double>rand() / RAND_MAX)
    la.uniform_sample_unit_sphere_inplace(polymer.direction)
    la.inplace_scale3(polymer.direction, amp)
    la.generate_translation_mat(polymer.direction, polymer)
    bound_0 = rand() % polymer.num_beads
    bound_1 = beads.from_point(
        amp_bead, polymer.num_beads, bound_0
    )
    ind0, indf = beads.check_bead_bounds(
        bound_0, bound_1, polymer.num_beads
    )
    inds = np.arange(ind0, indf)
    n_inds = indf - ind0
    polymer.last_amp_bead = n_inds
    polymer.last_amp_move = amp

    # "Efficient" Matrix Multiplication
    for i in range(n_inds):
        for j in range(3):
            polymer.r_trial[inds[i], j] = polymer.r[inds[i], j] +\
                polymer.transformation_mat[j, 3]
            polymer.t3_trial[inds[i], j] = polymer.t3[inds[i], j]
            polymer.t2_trial[inds[i], j] = polymer.t2[inds[i], j]

    return inds


# Tangent Rotation

cpdef long[:] tangent_rotation(
    poly.PolymerBase polymer, double amp_move, long amp_bead
):
    """Random bead rotation for a random selection of beads.
    
    Notes
    -----
    Select a random rotation angle. Then select some random number of beads 
    to rotate, requiring that at least one bead be selected if the move is 
    on. Call `rotate_selected_beads` to generate a random axis of rotation 
    for each bead, update the transformation matrix, and apply the rotation 
    to each selected bead.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object affected by tangent rotation
    amp_move : double
        Range of angles allowed for single tangent rotation move
    amp_bead : int
        Number of beads to randomly rotate

    Returns
    -------
    array_like (M,) of long
        Indices of M beads affected by the MC move
    """
    # Type Declarations
    cdef double ang
    cdef long num_beads_to_move, num_beads, n_inds
    cdef long[:] inds

    # Stochastic Move Selection
    ang = pick_random_amp(amp_move)
    num_beads_to_move = rand() % amp_bead + 1
    num_beads = polymer.num_beads
    inds = get_inds(num_beads, num_beads_to_move)
    n_inds = num_beads_to_move
    polymer.last_amp_bead = n_inds
    polymer.last_amp_move = ang
    
    # Rotate selected beads
    rotate_select_beads(polymer, inds, n_inds, ang)

    return inds


cdef void rotate_select_beads(
    poly.PolymerBase polymer, long[:] inds, long n_inds, double ang
):
    """Rotate the tangent vector of select beads about a random axis.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer affected by the MC move
    inds : long[:]
        Indices of beads affected by the MC move
    n_inds : long
        Number of beads involved in MC move
    ang : float
        Angle for which to randomly rotate tangents
    """
    cdef long num_points
    cdef double ele_t3, ele_t2
    cdef double[:] axis

    for i in range(n_inds):
        la.uniform_sample_unit_sphere_inplace(polymer.direction)
        la.arbitrary_axis_rotation(polymer.direction, origin, ang, polymer)
        for j in range(3):
            ele_t3 = 0
            ele_t2 = 0
            for k in range(3):
                ele_t3 += polymer.transformation_mat[j, k] *\
                    polymer.t3[inds[i], k]
                ele_t2 += polymer.transformation_mat[j, k] *\
                    polymer.t2[inds[i], k]
            polymer.t3_trial[inds[i], j] = ele_t3
            polymer.t2_trial[inds[i], j] = ele_t2


cdef long[:] get_inds(long num_beads, long num_inds):
    """Draw indices for the tangent rotation move.

    Parameters
    ----------
    num_beads : long
        Number of polymer beads from which to draw indices
    num_inds : long
        Number of indices to draw

    Returns
    -------
    array_like (M,) of long
        Array of M bead indices for which to apply tangent rotation move
    """
    cdef long ind0
    cdef bint redraw
    cdef long[:] inds
    cdef set inds_picked

    inds = np.empty(num_inds, dtype=int)
    inds_picked = set()
    for i in range(num_inds):
        redraw = True
        while redraw:
            ind0 = rand() % num_beads
            if ind0 not in inds_picked:
                inds[i] = ind0
                inds_picked.add(ind0)
                redraw = False
    return inds


# Full Chain Rotation

cpdef long[:] full_chain_rotation(
    poly.PolymerBase polymer, double amp_move, long amp_bead
):
    """Rotate an entire polymer about an arbitrary axis.
    
    Notes
    -----
    This move does not change a polymer's internal configurational (elastic)
    energy and is only relevant to simulations involving more than one polymer.

    The rotation takes place about a random axis, sampled uniformally from the
    unit sphere. The fulcrum of the rotation is a random bead of the polymer.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object
    amp_move : double
        Maximum amplitude (rotation angle) of the rotation move
    amp_bead : long
        Maximum amplitude of the window of beads selected for the move; this 
        attribute is included simply to match the formatting of the other 
        move functions and does not affect the full chain rotation 

    Returns
    -------
    array_like (N,) of long
        Indices of beads affected by the MC move (will always be entire polymer)
    """
    # Type Declarations
    cdef double ang, ele_r, ele_t3, ele_t2
    cdef long num_beads, n_inds, fulcrum_ind
    cdef long[:] inds
    cdef long[:, ::1] states
    cdef double[:] axis, fulcrum

    # Stochastic Move Selection
    n_inds = polymer.num_beads
    ang = pick_random_amp(amp_move)
    axis = la.uniform_sample_unit_sphere()
    polymer.last_amp_bead = n_inds
    polymer.last_amp_move = ang
    
    # Prep for Deterministic Transformation
    fulcrum_ind = rand() % n_inds
    fulcrum = np.empty((3,), dtype='d')
    for i in range(3):
        fulcrum[i] = polymer.r[fulcrum_ind, i]
    la.arbitrary_axis_rotation(axis, fulcrum, ang, polymer)

    # "Efficient" matrix multiplication
    for i in range(n_inds):
        for j in range(3):
            ele_r = 0
            ele_t3 = 0
            ele_t2 = 0
            for k in range(3):
                ele_r += polymer.transformation_mat[j, k] * polymer.r[i, k]
                ele_t3 += polymer.transformation_mat[j, k] * polymer.t3[i, k]
                ele_t2 += polymer.transformation_mat[j, k] * polymer.t2[i, k]
            polymer.r_trial[i, j] = ele_r + polymer.transformation_mat[j, 3]
            polymer.t3_trial[i, j] = ele_t3
            polymer.t2_trial[i, j] = ele_t2

    return polymer.all_inds


# Full Chain Translation

cpdef long[:] full_chain_translation(
    poly.PolymerBase polymer,
    double amp_move,
    long amp_bead
):
    """
    Translate an entire polymer in an arbitrary direction.
    
    Notes
    -----
    This move does not change a polymer's internal configurational (elastic)
    enregy and is only relevant to simulations involving more than one polymer.

    The translation occurs in a random direction, sampled uniformally from a
    unit sphere.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object
    amp_move : double
        Maximum amplitude (translation distance) of the slide move
    amp_bead : long
        Maximum amplitude of the window of beads selected for the move; this 
        attribute is included simply to match the formatting of the other 
        move functions and does not affect the full chain translation

    Returns
    -------
    array_like (N,) of long
        Indices of beads affected by the MC move (will always be entire polymer)
    """
    # Type Declarations
    cdef double amp, dx, dy, dz
    cdef long num_beads, n_inds, i, j
    cdef long[:] inds
    cdef long[:, ::1] states
    cdef double[:] direction, delta
    cdef double[:, ::1] translation_mat, r, t3, t2, r_try
    cdef double[::1, :] r_try_T, t3_T, t2_T

    # Stochastic Move Selection
    amp = amp_move * (<double>rand() / RAND_MAX)
    direction = la.uniform_sample_unit_sphere()
    delta = la.vec_scale3(direction, amp)
    la.generate_translation_mat(delta, polymer)
    n_inds = polymer.num_beads
    polymer.last_amp_bead = n_inds
    polymer.last_amp_move = amp

    # "Efficient" Matrix Multiplication
    for i in range(n_inds):
        for j in range(3):
            polymer.r_trial[i, j] = polymer.r[i, j] +\
                polymer.transformation_mat[j, 3]

    return polymer.all_inds


# Change Binding States

cpdef long[:] change_binding_state(
    poly.PolymerBase polymer, double amp_move, long amp_bead
):
    """Flip the binding state of a reader protein.
    
    Notes
    -----
    Before applying the move, check that the polymer has any reader proteins at
    all.

    Begin the move by identifying the number of binding sites that each bead
    has for the particular reader protein. Then randomly select a bead in the
    chain. Select a second bead from a two sided decaying exponential
    distribution around the index of the first bead. Replace the binding state
    of the selected beads.

    In our model, we do not track the binding state of individual tails. We
    care only about how many tails are bound. Therefore, for each move, we will
    generate a random order of bound and unbound tails and will flip the state
    of the first M tails in that order.

    This function assumes that polymers are formed from a single type of bead
    object. if different types of bead objects exist, then the first two
    if-statements that reference `bead[0]` will need to be generalized.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object on which the move is applied
    amp_move : double
        Magnitude of the MC move; this attribute is included simply for 
        consistency with the other MC move functions and does not affect the 
        binding state move
    amp_bead : int
        Maximum range of beads to which a binding state swap will take palce

    Returns
    -------
    array_like (M,) of long
        Indices of M beads affected by the MC move
    """
    # Type Declarations
    cdef long binder_ind, n_tails, n_flip, b_0, b_1, ind0, indf, n_inds
    cdef long[:] inds, possible_flips

    # Stochastic Move Selection
    binder_ind = rand() % polymer.num_binders
    n_tails = polymer.beads[0].binders[binder_ind].sites_per_bead
    b_0 = rand() % polymer.num_beads
    b_1 = beads.from_point(amp_bead, polymer.num_beads, b_0)
    ind0, indf = beads.check_bead_bounds(b_0, b_1, polymer.num_beads)
    inds = np.arange(ind0, indf)
    n_inds = indf - ind0
    possible_flips = np.arange(1, n_tails + 1, 1)
    num_prob_bounds = n_tails - 1
    n_flip = rand() % (n_tails + 1)
    polymer.last_amp_bead = n_inds
    polymer.last_amp_move = <double> n_flip

    # Perform Binding Move
    conduct_change_binding_states(
        polymer, ind0, indf, n_inds, inds, binder_ind, n_tails, n_flip
    )
    return inds


cdef void conduct_change_binding_states(
    poly.PolymerBase polymer, long ind0, long indf, long n_inds, long[:] inds,
    long binder_ind, long num_tails, long num_tails_flipped
):
    """Get new binding states for the beads affected by the binding state move.

    Parameters
    ----------
    polymer : poly.PolymerBase
        Polymer object on which the binding state move is applied
    ind0, indf : long
        First bead index and one past the last bead index affected by move
    n_inds : long
        Number of beads affected by binding state move (equal to `indf`-`ind0`)
    inds : array_like (M,)
        Bead indices in the polymer to which the change binding state move is
        applied
    binder_ind : long
        Index of the binder for which the state is being swapped; the value
        represents the column of the states array affected by the move
    num_tails : long
        Number of binding sites of the bead for the particular binder being
        flipped by the move.
    num_tails_flipped : long
        Number of binding sites on the bead to flip
    """
    cdef long i

    for i in range(n_inds):
        polymer.states_trial[inds[i], binder_ind] = get_new_state(
            polymer.beads[0].binders[binder_ind],
            polymer.states[inds[i], binder_ind],
            num_tails, num_tails_flipped
        )


cdef long get_new_state(
    Binder binder, long state, long num_tails, long num_tails_flipped
):
    """Get a next binding state of a bead.

    Parameters
    ----------
    binder : Binder
        The Binder object whose binding state is being changed
    state : long
        Current binding state of the bead – how many bead tails are bound
    num_tails : long
        Number of tails for the particular binder which may be bound
    num_tails_flipped : long
        Number of tails for the particular binder which are swapped

    Returns
    -------
    long
        Number of tails that are bound after the move
    """
    cdef long new_binding_state
    new_binding_state = np.random.randint(0, num_tails + 1)
    return new_binding_state
