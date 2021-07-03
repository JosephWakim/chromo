"""Functions for performing Monte Carlo transformations.
"""

# Built-in Modules
from typing import Tuple, List, Callable

# External Modules
import numpy as np
from numba import njit

# Custom Modules
import chromo.util.bead_selection as beads
import chromo.util.linalg as la
from chromo.polymers import Polymer
from chromo.marks import Epigenmark


MOVE_AMP = float
BEAD_AMP = int

_proposal_arg_names: List[str] = [
    'inds', 'r', 't3', 't2', 'states', 'continuous_inds', 'bead_amp',
    'move_amp'
]

_proposal_arg_types = Tuple[
    Tuple[int, int],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
    List[Epigenmark],
    bool,
    BEAD_AMP,
    MOVE_AMP
]

_move_arg_types = [
    Polymer,
    MOVE_AMP,
    BEAD_AMP
]

_move_func_type = Callable[[Polymer, MOVE_AMP, BEAD_AMP], _proposal_arg_types]


# Crank Shaft Move

def crank_shaft(
    polymer: Polymer,
    amp_move: MOVE_AMP,
    amp_bead: BEAD_AMP
) -> _proposal_arg_types:
    """
    Rotate section of polymer around axis formed by two end bead positions.

    The crank shaft move affects a continuous range of beads, and the internal
    configuration is unaffected by the crank shaft move. Therefore, when
    evaluating polymer energy change associated with the move, it is sufficent
    to look only at the ends of the affected segment.

    Begin by randomly selecting a starting and ending index for the crank-shaft
    move based on the bead amplitude. Then generate a random rotation angle for
    the move based on the move amplitude. Obtain the axis of rotation from the
    change in position between the starting and ending beads. Finally, generate
    the rotation matrix and obtain trial positions and tangents for evaluation.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (rotation angle) of the crank-shaft move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the crank-shaft move

    Returns
    -------
    _proposal_arg_types
        Affected bead indices, their positions and orientations, an
        indicator that a continuous segment of beads were affected,
        and the move and bead amplitudes
    """
    rot_angle: float = amp_move * (np.random.rand() - 0.5)
    num_beads: int = polymer.num_beads
    bound_0: int = np.random.randint(num_beads)
    bound_1: int = max(int(beads.from_point(amp_bead, num_beads, bound_0)), 1)
    ind0, indf = beads.check_bead_bounds(bound_0, bound_1, num_beads)
    inds: np.ndarray = np.arange(ind0, indf)
    continuous_inds: bool = True

    axis: np.ndarray = get_crank_shaft_axis(polymer, ind0, indf)
    r_points: np.ndarray = np.ones((4, indf-ind0))
    r_points[0:3] = polymer.r[inds].T
    t3_points: np.ndarray = np.zeros((4, indf-ind0))
    t3_points[0:3] = polymer.t3[inds].T
    t2_points: np.ndarray = np.zeros((4, indf-ind0))
    t2_points[0:3] = polymer.t2[inds].T

    point: np.ndarray = r_points[0:3, 0].flatten()
    rot_mat: np.ndarray = la.arbitrary_axis_rotation(axis, point, rot_angle)
    r_trial, t3_trial, t2_trial = conduct_crank_shaft(
        r_points, t3_points, t2_points, rot_mat
    )
    r_trial = r_trial[0:3].T
    t3_trial = t3_trial[0:3].T
    t2_trial = t2_trial[0:3].T

    return (
        inds, r_trial, t3_trial, t2_trial, None, continuous_inds, len(inds),
        rot_angle
    )


def get_crank_shaft_axis(
    polymer: Polymer,
    ind0: int,
    indf: int
) -> np.ndarray:
    """Get the axis of rotation for the crank shaft move from bead selection.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    ind0 : int
        Starting bead index for crank-shaft move
    indf : int
        One past ending bead index for crank-shaft move

    Returns
    -------
    np.ndarray (3,)
        Normalized vector giving orientation of rotation axis
    """
    if ind0 == (indf - 1) and ind0 == 0:
        axis = polymer.r[ind0+1] - polymer.r[ind0]
    elif ind0 == (indf - 1) and ind0 == len(polymer.r)-1:
        axis = polymer.r[len(polymer.r)-1] - polymer.r[len(polymer.r)-2]
    elif ind0 == (indf - 1):
        axis = polymer.r[ind0+1] - polymer.r[ind0-1]
    else:
        axis = polymer.r[indf - 1] - polymer.r[ind0]
    return axis / np.linalg.norm(axis)


def conduct_crank_shaft(
    r_points: np.ndarray,
    t3_points: np.ndarray,
    t2_points: np.ndarray,
    rot_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform deterministic operation of crank_shaft move.

    Begin by determining the change in tangent vector orientation from the
    start to end of the affected beads. Then isolate the coordinates of the
    starting bead. Define a rotation matrix, specify a rotation vector, and
    generate trial positions and orientations.

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing rotation.
    t3_points, t2_points : array_like (4, N)
        Homogeneous tangent vectors for beads undergoing rotation
    rot_mat : array_like (4, 4)
        Homogeneous rotation matrix for a specified angle of rotation, with
        a positive rotation angle defined as counterclockwise with rotation
        axis pointed directly out of screen

    Returns
    -------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following rotation
    t3_trial, t2_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    """
    r_trial = conduct_rotation(r_points, rot_mat)
    t3_trial = conduct_rotation(t3_points, rot_mat)
    t2_trial = conduct_rotation(t2_points, rot_mat)
    return r_trial, t3_trial, t2_trial


@njit
def conduct_rotation(matrix: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    """Rotate arbitrary homogeneous matrix using rotation matrix

    Parameters
    ----------
    matrix : array_like (4, N)
        Homogeneous matrix of points being rotated
    rot_mat : array_like (4, 4)
        Homogeneous rotation matrix

    Returns
    -------
    array_like (4, N)
        Array of homogeneous vectors after rotation by rotation matrix
    """
    return rot_mat @ matrix


# End Pivot Move

def end_pivot(
    polymer: Polymer,
    amp_move: MOVE_AMP,
    amp_bead: BEAD_AMP
) -> _proposal_arg_types:
    """
    Randomly rotate segment from end of polymer about random axis.

    Stochastic component of end-pivot move. Begin by selecting a random
    rotation angle based on the move amplitude. Then isolate a random selection
    of beads on either the LHS or RHS of the polymer based on the move
    amplitude. Isolate the homogeneous coordinates for the selected beads, and
    format those coordinates for matrix multiplication. Perform the
    transformation associated with the end pivot move, with random axis of
    rotation selected along the unit sphere. Reformat homogeneous coordinates
    as cartesian coordinates.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (rotation angle) of the end-pivot move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the end-pivot move

    Returns
    -------
    _proposal_arg_types
        Affected bead indices, their positions and orientations, and an
        indicator that a continuous segment of beads were affected
    """
    rot_angle = amp_move * (np.random.rand() - 0.5)
    num_beads = polymer.num_beads
    rotate_LHS = (np.random.randint(0, 2) == 0)

    if rotate_LHS:
        ind0 = 0
        indf = beads.from_left(amp_bead, num_beads) + 1
        indf = min(indf, num_beads)
    else:
        ind0 = beads.from_right(amp_bead, num_beads)
        ind0 = min(ind0, num_beads-1)
        indf = num_beads
    inds = np.arange(ind0, indf)
    continuous_inds = True

    axis = la.uniform_sample_unit_sphere()
    fulcrum = polymer.r[indf-1] if rotate_LHS else polymer.r[ind0]
    rot_mat = la.arbitrary_axis_rotation(axis, fulcrum, rot_angle)
    r_points = np.ones((4, indf-ind0))
    r_points[0:3] = polymer.r[inds].T
    t3_points = np.zeros((4, indf-ind0))
    t3_points[0:3] = polymer.t3[inds].T
    t2_points = np.zeros((4, indf-ind0))
    t2_points[0:3] = polymer.t2[inds].T

    r_trial, t3_trial, t2_trial = conduct_end_pivot(
        r_points, t3_points, t2_points, rot_mat
    )
    r_trial = r_trial[0:3].T
    t3_trial = t3_trial[0:3].T
    t2_trial = t2_trial[0:3].T

    return (
        inds, r_trial, t3_trial, t2_trial, None, continuous_inds, len(inds),
        rot_angle
    )


@njit
def conduct_end_pivot(
    r_points: np.ndarray,
    t3_points: np.ndarray,
    t2_points: np.ndarray,
    rot_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotation of fixed sub set of beads.

    Deterministic component of end-pivot move. Generate trial coordinates and
    two orthogonal tangent vectors.

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing rotation.
    t3_points : array_like (4, N)
        Homogeneous tangent vectors for beads undergoing rotation
    t2_points : array_like (4, N)
        Homogeneous tangent, orthogonal to t3 tangents, for rotating beads
    rot_mat : array_like (4, 4)
        Homogeneous rotation matrix for a specified angle of rotation, with
        a positive rotation angle defined as counterclockwise with rotation
        axis pointed directly out of screen

    Returns
    -------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following rotation
    t3_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    t2_trial : array_like (4, N)
        Homogeneous tangent vectors, orthogonal to t3_trial, following rotation
    """
    r_trial = rot_mat @ r_points
    t3_trial = rot_mat @ t3_points
    t2_trial = rot_mat @ t2_points
    return r_trial, t3_trial, t2_trial


# Slide Move

def slide(
    polymer: Polymer,
    amp_move: MOVE_AMP,
    amp_bead: BEAD_AMP
) -> _proposal_arg_types:
    """
    Randomly translate a segment of the polymer.

    Randomly set the translation distance based on the slide move amplitude.
    Then randomly pick a direction by sampling uniformally from a unit sphere.
    Split the translation distance into x, y, z components using the direction.
    Select a random segment of beads to move based on the bead amplititude,
    and identify ordered start and end indices. Generate a matrix of homologous
    coordinates for the sliding beads, then apply the translation operation to
    identify final bead coordinates.

    NOTE: There is a difference in how we define the move amplitude between
    this code and the original FORTRAN codebase. In the original codebase, the
    move amplitude specifies the maximum translation in each dimension, while
    in this code, the move amplitude specifies the maximum magnitude of
    translation.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (translation distance) of the slide move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the slide move

    Returns
    -------
    _proposal_arg_types
        Affected bead indices, their positions and orientations, and an
        indicator that a continuous segment of beads were affected
    """
    translation_amp = amp_move * (np.random.rand())
    direction = la.uniform_sample_unit_sphere()
    slide_x, slide_y, slide_z = direction * translation_amp
    translation_mat = la.generate_translation_mat(slide_x, slide_y, slide_z)

    num_beads = polymer.num_beads
    bound_0 = np.random.randint(num_beads)
    bound_1 = max(
        int(beads.from_point(amp_bead, num_beads, bound_0)), 1
    )
    ind0, indf = beads.check_bead_bounds(bound_0, bound_1, num_beads)
    inds = np.arange(ind0, indf)
    continuous_inds = True

    r_points = np.ones((4, indf-ind0))
    r_points[0:3] = polymer.r[inds].T
    r_trial = conduct_slide(r_points, translation_mat)
    r_trial = r_trial[0:3].T

    return (
        inds, r_trial, None, None, None, continuous_inds, len(inds),
        translation_amp
    )


@njit
def conduct_slide(
    r_points: np.ndarray, translation_mat: np.ndarray
) -> np.ndarray:
    """
    Deterministic component of slide move.

    Conduct the slide move on set of beads with homogeneous coordinates stored
    in r_points.

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing slide move
    translation_mat : array_like (4, 4)
        Homogeneous translation matrix with which to translate r_points

    Return
    ------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following slide move

    """
    return translation_mat @ r_points


# Tangent Rotation

def tangent_rotation(
    polymer: Polymer,
    amp_move: MOVE_AMP,
    amp_bead: BEAD_AMP
) -> _proposal_arg_types:
    """
    Random bead rotation for a random selection of beads.

    Calls `one_bead_tangent_rotation` for each randomly selected bead to
    undergo rotation.

    Begins by selecting some number of beads to undergo rotation based on the
    bead amplitude, requiring that at least one bead be selected if the move
    is on. Select a random rotation angle and loop through the number of beads
    undergoing the rotation; the `one_bead_tangent_rotation` method handles
    selection of a random bead index to undergo the rotation. The position and
    tangent vectors of the affected beads are then reformatted, sorted, and
    evaluated for energy change.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Range of angles allowed for single tangent rotation move
    amp_bead : int
        Number of beads to randomly rotate
    """
    rot_angle = amp_move * (np.random.rand() - 0.5)
    num_beads_to_move = max(int(np.random.uniform() * amp_bead), 1)
    num_beads = polymer.num_beads
    inds = get_inds(num_beads, num_beads_to_move)
    t3_inds = np.ones((4, num_beads_to_move))
    t3_inds[0:3, :] = polymer.t3[inds].T
    t2_inds = np.ones((4, num_beads_to_move))
    t2_inds[0:3, :] = polymer.t2[inds].T
    t3_trial, t2_trial = rotate_select_beads(
        t3_inds, t2_inds, rot_angle
    )
    t3_trial = t3_trial[0:3, :].T
    t2_trial = t2_trial[0:3, :].T
    continuous_inds = False
    return (
        inds, None, t3_trial, t2_trial, None, continuous_inds, len(inds),
        rot_angle
    )


def rotate_select_beads(
    t3_points: np.ndarray,
    t2_points: np.ndarray,
    rot_angle: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the tangent vector of select beads about a random axis.

    Parameters
    ----------
    t3_points : array_like (4, N)
        Homogeneous tangent vectors for beads undergoing rotation
    t2_points : array_like (4, N)
        Homogeneous tangent, orthogonal to t3 tangents, for rotating beads
    rot_angle : array_like (4, N)
        Angle for which to randomly rotate tangents

    Returns
    -------
    array_like (4, N)
        Homogeneous array of t3 tangent vectors after tangent rotation is
        applied
    array_like (4, N)
        Homogeneous array of t2 tangent vectors after tangent rotation is
        applied
    """
    origin = np.array([0, 0, 0])
    num_points = t3_points.shape[1]
    if t2_points.shape[1] != num_points:
        raise ValueError(
            "Dimensions of tangent arrays in `rotate_select_beads` is \
            inconsistent."
        )

    t3_trial = np.empty(t3_points.shape)
    t2_trial = np.empty(t2_points.shape)
    for i in range(num_points):
        axis = la.uniform_sample_unit_sphere()
        rot_mat = la.arbitrary_axis_rotation(
            axis, origin, rot_angle
        )
        t3_trial[:, i] = conduct_rotation(t3_points[:, i], rot_mat)
        t2_trial[:, i] = conduct_rotation(t2_points[:, i], rot_mat)

    return t3_trial, t2_trial


def get_inds(num_beads: int, num_inds: int) -> np.ndarray:
    """Draw indices for the tangent rotation move.

    Parameters
    ----------
    num_beads : int
        Number of polymer beads from which to draw indices
    num_inds : int
        Number of indices to draw

    Returns
    -------
    array_like (N, )
        Array of bead indices for which to apply tangent rotation move
    """
    inds = np.ones(num_inds, dtype=int) * -1
    for i in range(num_inds):
        redraw: bool = True
        while redraw:
            ind0: int = np.random.randint(num_beads)
            if ind0 not in inds:
                inds[i] = ind0
                redraw = False
    inds = np.sort(inds)
    return inds


# Full Chain Rotation

def full_chain_rotation(
    polymer: Polymer,
    amp_move: MOVE_AMP,
    **kwargs
) -> _proposal_arg_types:
    """
    Rotate an entire polymer about an arbitrary axis.

    This move does not change a polymer's internal configurational enregy and
    is only relevant to simulations involving more than one polymer.

    The rotation takes place about a random axis, sampled uniformally from the
    unit sphere. The fulcrum of the rotation is a random bead of the polymer.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (translation distance) of the slide move

    Returns
    -------
    _proposal_arg_types
        Affected bead indices, their positions and orientations, and an
        indicator that a continuous segment of beads were affected
    """
    num_beads = len(polymer.r)
    inds = np.arange(0, num_beads, 1)
    continuous_inds = True
    rot_angle = amp_move * (np.random.rand() - 0.5)
    axis = la.uniform_sample_unit_sphere()
    fulcrum = polymer.r[np.random.randint(0, num_beads)]
    rot_mat = la.arbitrary_axis_rotation(axis, fulcrum, rot_angle)

    r = np.ones((4, num_beads))
    r[0:3, :] = polymer.r.T
    t3 = np.ones((4, num_beads))
    t3[0:3, :] = polymer.t3.T
    t2 = np.ones((4, num_beads))
    t2[0:3, :] = polymer.t2.T

    r_trial, t3_trial, t2_trial = conduct_end_pivot(r, t3, t2, rot_mat)
    r_trial = r_trial[0:3].T
    t3_trial = t3_trial[0:3].T
    t2_trial = t2_trial[0:3].T
    return (
        inds, r_trial, t3_trial, t2_trial, None, continuous_inds, num_beads,
        rot_angle
    )


# Full Chain Translation

def full_chain_translation(
    polymer: Polymer,
    amp_move: MOVE_AMP,
    **kwargs
) -> _proposal_arg_types:
    """
    Translate an entire polymer in an arbitrary direction.

    This move does not change a polymer's internal configurational enregy and
    is only relevant to simulations involving more than one polymer.

    The translation occurs in a random direction, sampled uniformally from a
    unit sphere.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (translation distance) of the slide move

    Returns
    -------
    _proposal_arg_types
        Affected bead indices, their positions and orientations, and an
        indicator that a continuous segment of beads were affected
    """
    num_beads = len(polymer.r)
    inds = np.arange(0, num_beads, 1)
    continuous_inds = True
    translation_amp = amp_move * (np.random.rand())
    direction = la.uniform_sample_unit_sphere()
    slide_x, slide_y, slide_z = direction * translation_amp
    translation_mat = la.generate_translation_mat(slide_x, slide_y, slide_z)

    r = np.ones((4, num_beads))
    r[0:3, :] = polymer.r.T
    r_trial = conduct_slide(r, translation_mat)
    r_trial = r_trial[0:3].T

    return (
        inds, r_trial, None, None, None, continuous_inds, None,
        translation_amp
    )


# Change Binding States

def change_binding_state(
    polymer: Polymer,
    *
    amp_bead: BEAD_AMP,
    **kwargs
) -> _proposal_arg_types:
    """Flip the binding state of a mark.

    Before applying the move, check that the polymer has any marks at all.

    Begin the move by identifying the number of binding sites that each bead
    has for the particular mark. Then randomly select a bead in the chain.
    Select a second bead from a two sided decaying exponential distribution
    around the index of first. Replace the binding state of the selected beads.

    In our model, we do not track the binding state of individual tails. We
    care only about how many tails are bound. Therefore, for each move, we will
    generate a random order of bound and unbound tails and will flip the state
    of the first M tails in that order.

    NOTE: This function assumes that polymers are formed from a single type of
    bead object. if different types of bead objects exist, then the first two
    if-statements that reference `bead[0]` will need to be generalized.

    Parameters
    ----------
    polymer : Polymer
        Polymer object on which the move is applied
    mark_ind : int
        Index of marks bound to the polymer on which the swap will take place
    amp_bead : BEAD_AMP
        Maximum range of beads to which a binding state swap will take palce

    Returns
    -------
    _proposal_arg_types
        Affected bead indices, their positions and orientations, and an
        indicator that a continuous segment of beads were affected
    """
    if polymer.beads[0].mark_names is None:
        return None, None, None, None, None, None, None, None

    num_marks = len(polymer.beads[0].mark_names)
    mark_ind = np.random.choice(np.arange(num_marks))
    num_tails = polymer.beads[0].marks[mark_ind].sites_per_bead
    num_beads = polymer.num_beads
    bound_0 = np.random.randint(num_beads)
    bound_1 = max(
        int(beads.from_point(amp_bead, num_beads, bound_0)), 1
    )
    ind0, indf = beads.check_bead_bounds(bound_0, bound_1, num_beads)
    inds = np.arange(ind0, indf)
    continuous_inds = True
    possible_flips = np.arange(1, num_tails + 1, 1)
    prob_bounds = [
        1 / possible_flips * (i + 1) for i in range(len(possible_flips) - 1)
    ]
    rand_val = np.random.random()
    for i in range(len(prob_bounds)):
        if rand_val > i:
            num_tails_flipped = possible_flips[i]
            break
    states = conduct_change_binding_states(
        polymer, inds, mark_ind, num_tails, num_tails_flipped
    )
    return inds, None, None, None, states, continuous_inds, amp_bead, None


def conduct_change_binding_states(
    polymer: Polymer,
    inds: np.ndarray,
    mark_ind: int,
    num_tails: int,
    num_tails_flipped: int
) -> np.ndarray:
    """Deterministic component of the change binding state move.

    Parameters
    ----------
    polymer : Polymer
        Polymer object on which the move is applied
    inds : np.ndarray (M, )
        Bead indices in the polymer to which the change binding state move is
        applied
    mark_ind : int
        Index of the mark for which the state is being swapped; the value
        represents the column of the states array affected by the move
    num_tails : int
        Number of binding sites of the bead for the particular mark being
        flipped by the move.
    num_tails_flipped : int
        Number of binding sites on the bead to flip

    Returns
    -------
    np.ndarray (M, N)
        Array of binding states for the all N marks (not just the swapped mark)
        at the M swapped beads
    """
    states = polymer.states[inds, :]
    for i in range(len(inds)):
        states[i, mark_ind] = get_new_state(
            states[i, mark_ind], num_tails, num_tails_flipped
        )
    return states


def get_new_state(state: int, num_tails: int, num_tails_flipped: int) -> int:
    """Get a next binding state of a bead.

    Parameters
    ----------
    state : int
        Current binding state of the bead – how many bead tails are bound
    num_tails : int
        Number of tails for the particular mark which may be bound
    num_tails_flipped : int
        Number of tails for the particular mark which are swapped

    Returns
    -------
    int
        Number of tails that are marked after the move
    """
    binding_seq = np.random.shuffle(
        np.array([1] * state + [0] * (num_tails - state))
    )
    for i in range(num_tails_flipped):
        binding_seq[i] = (binding_seq[i] + 1) % 2
    return np.sum(binding_seq)
