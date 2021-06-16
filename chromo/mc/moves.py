"""Utilities for proposing Monte Carlo moves.

This module include an MC adapter class which will drive move adaption and
functions applying transformations made by each move.
"""

# Built-in Modules
# from time import process_time
from typing import Tuple, List, Callable, Optional

# External Modules
import numpy as np

# Custom Modules
import chromo.util.bead_selection as beads
import chromo.util.linalg as linalg
import chromo.mc.mc_stat as mc_stat
from chromo.components import Polymer
from chromo.marks import Epigenmark


MOVE_AMP = float
BEAD_AMP = int

_proposal_arg_names = [
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

_move_func_type = Callable[
    [Polymer, MOVE_AMP, BEAD_AMP], _proposal_arg_types
]


class MCAdapter:
    """Track success rate and adjust parameters for Monte Carlo moves.

    In order to ensure that a Monte Carlo simulation equilibrates as quickly as
    possible, we provide an automated system for tuning the "aggressiveness" of
    each Monte Carlo move in order to optimize the tradeoff between

    1. The move being large enough that it does not take too many moves to
       equilibrate, and
    2. The move not being so large that it will never be
       accepted, or the computational cost becomes prohibitive (especially in
       the case when the computational cost scales super-linearly).

    All Monte-Carlo moves we currently use for our polymers have two
    "amplitudes".  The "bead" amplitude controls how many beads are typically
    affected by the move, and the "move" amplitude controls how "far" those
    beads are moved.  Typically, actually bead counts and move sizes are
    selected randomly, so these amplitudes will correspond to averages.

    Success rate will be tracked by a `PerformanceTracker` object specified in
    the `chromo/mc/mc_stat.py` module. The performance tracker will indicate an
    overall acceptance rate based on the total numbers of moves attempted and
    accepted. A running acceptance rate will also be maintained, which decays
    the weight of historic values using a decay factor.

    In the future, we want to do this optimization by looking at actual metrics
    based on the simulation output, but for now only implementation of an
    MCAdapter (`MCAdaptSuccessRate`) handles this optimization by simply trying
    to maintain a fixed target success rate.
    """
    def __init__(
        self,
        log_path: str,
        move_func: _move_func_type,
        decay_rate: Optional[float] = 0.95,
        init_amp_bead: Optional[int] = 100,
        init_amp_move: Optional[float] = 0.05,
    ):
        """Initialize the MCAdapter object.

        Parameters
        ----------
        log_path : str
            Path to log file tracking bead/move amplitudes and acceptance rate
        move_func : Callable[_move_arg_types, _proposal_arg_types]
            Functions representing Monte Carlo moves
        decay_rate : Optional[float]
            Factor by which to downweight historical acceptance rate in mean
            (default = 0.95)
        init_amp_bead : Optional[int]
            Initial bead selection amplitude (default = 100)
        init_amp_move : Optional[float]
            Initial move amplitude (default = 0.05)
        """
        self.name = move_func.__name__
        self.move_func = move_func
        self.amp_move = init_amp_move
        self.num_per_cycle = 1
        self.amp_bead = init_amp_bead
        self.num_attempt = 0
        self.num_success = 0
        self.move_on = True
        self.last_amp_move = None
        self.last_amp_bead = None

        self.performance_tracker =\
            mc_stat.PerformanceTracker(log_path, decay_rate)

    def __str__(self):
        return f"MCAdapter<{self.name}>"

    def to_file(self, path):
        pass

    def propose(self, polymer):
        """
        Get new proposed state of the system.

        Parameters
        ----------
        polymer : `chromo.Polymer`
            The polymer to attempt to move.

        Returns
        -------
        inds : array_like (N,)
            List of indices for N beads being moved
        r : (N, 3) array_like of float, optional
            The proposed new positions of the moved beads. Throughout this
            method, ``N = indf - ind0 + 1`` is the number of moved beads.
        t3 : (N, 3) array_like of float, optional
            The proposed new tangent vectors.
        t2 : (N, 3) array_like of float, optional
            Proposed new material normals.
        states : (N, M) array_like of int, optional
            The proposed new chemical states, where *M* is the number of
            chemical states associated with the given polymer.
        """
        self.num_attempt += 1
        return self.move_func(
            polymer=polymer, amp_move=self.amp_move, amp_bead=self.amp_bead
        )

    @staticmethod
    def replace_none(poly, *proposal):
        """
        Fill in empty parts of proposed move with original polymer state.

        Parameters
        ----------
        poly : Polymer
            The Polymer object being modified by the move.
        *proposal : inds, r, t3, t2, states, continuous_inds
            The proposed Monte Carlo move, where each of r, t3, t2, states are
            an ``Optional[np.ndarray]``.

        Returns
        -------
        r, t3, t2, states : Tuple[np.ndarray<M,3>]
        """
        prop_names = _proposal_arg_names
        kwargs = {prop_names[i]: proposal[i] for i in range(len(prop_names)-2)}

        # Remove inds and continuous_inds from proposal
        inds = kwargs.pop('inds')
        kwargs.pop("continuous_inds")
        prop_names = prop_names[1:len(prop_names)-3]
        actual_proposal = []
        for i, name in enumerate(prop_names):
            prop = kwargs[name]
            if prop is not None:
                actual_proposal.append(prop)
            else:
                prop = []
                for ind in inds:
                    prop.append(poly.__dict__[name][ind])
                actual_proposal.append(prop)

        return actual_proposal

    def accept(self, poly, dE, *proposal):
        """Update polymer with new state and update proposal stats.

        Update all elements of `poly` for which proposed state is not None. Log
        the move acceptance/rejection in the move performance tracker.

        Parameters
        ----------
        poly : Polymer
            Polymer affected by the MC move
        dE : float
            Change in energy associated with the move
        *proposal : _proposal_arg_types
            New states proposed by the MC move
        """
        inds = proposal[0]
        r, t3, t2, states = MCAdapter.replace_none(poly, *proposal)
        for i in range(len(inds)):
            poly.beads[inds[i]].r = r[i]
            poly.beads[inds[i]].t3 = t3[i]
            poly.beads[inds[i]].t2 = t2[i]
            poly.beads[inds[i]].states = states[i]
            poly.r[inds[i]] = r[i]
            poly.t3[inds[i]] = t3[i]
            poly.t2[inds[i]] = t2[i]
            poly.states[inds[i]] = states[i]

        self.num_success += 1
        self.performance_tracker.update_acceptance_rate(accept=True)
        self.performance_tracker.log_move(
            self.amp_move,
            self.amp_bead,
            self.last_amp_move,
            self.last_amp_bead,
            dE
        )
        self.performance_tracker.log_acceptance_rate()

    def reject(self, dE):
        """Reject a proposed Monte Carlo move.

        Log the rejected move in the performance tracker

        Parameters
        ----------
        dE : float
            Change in energy associated with the move
        """
        self.performance_tracker.update_acceptance_rate(accept=False)
        self.performance_tracker.log_move(
            self.amp_move,
            self.amp_bead,
            self.last_amp_move,
            self.last_amp_bead,
            dE
        )
        self.performance_tracker.log_acceptance_rate()


# Specify Monte Carlo Moves
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
    # print("Crank-shaft move")
    # start = process_time()
    rot_angle = amp_move * (np.random.rand() - 0.5)

    num_beads = polymer.num_beads
    bound_0 = np.random.randint(num_beads)
    bound_1 = max(
        int(beads.select_bead_from_point(amp_bead, num_beads, bound_0)), 1
    )
    ind0, indf = beads.check_bead_bounds(bound_0, bound_1, num_beads)
    inds = np.arange(ind0, indf)
    continuous_inds = True

    axis = get_crank_shaft_axis(polymer, ind0, indf)
    r_points = np.ones((4, indf-ind0))
    r_points[0:3] = polymer.r[inds].T
    t3_points = np.zeros((4, indf-ind0))
    t3_points[0:3] = polymer.t3[inds].T
    r_trial, t3_trial = conduct_crank_shaft(
        r_points, t3_points, axis, rot_angle
    )
    r_trial = r_trial[0:3].T
    t3_trial = t3_trial[0:3].T
    # print(process_time() - start)

    return (
        inds, r_trial, t3_trial, None, None, continuous_inds, len(inds),
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
    axis: np.ndarray,
    rot_angle: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform deterministic operation of crank_shaft move.

    Begin by determining the change in tangent vector orientation from the
    start to end of the affected beads. Then isolate the coordinates of the
    starting bead. Define a rotation matrix, specify a rotation vector, and
    generate trial positions and orientations.

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing rotation.
    t3_points : array_like (4, N)
        Homogeneous tangent vectors for beads undergoing rotation
    axis : array_like (3,)
        Rotation axis
    rot_angle : float
        Angle of rotation. Positive rotation is counterclockwise when
        rotation axis is pointing directly out of the screen

    Returns
    -------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following rotation
    t3_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    """
    r_trial = np.ones(r_points.shape)
    t3_trial = np.zeros(r_points.shape)

    for i in range(r_points.shape[1]):
        r_ind = np.dot(r_points[0:3, i], axis) * axis
        rot_matrix = linalg.arbitrary_axis_rotation(axis, r_ind, rot_angle)
        r_trial[:, i] = rot_matrix @ r_points[:, i]
        t3_trial[:, i] = rot_matrix @ t3_points[:, i]

    return r_trial, t3_trial


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
    # print("End-pivot move")
    # start = process_time()
    rot_angle = amp_move * (np.random.rand() - 0.5)
    num_beads = polymer.num_beads
    rotate_LHS = (np.random.randint(0, 2) == 0)

    if rotate_LHS:
        ind0 = 0
        indf = beads.select_bead_from_left(
            amp_bead, num_beads
        ) + 1
        indf = min(indf, num_beads)
    else:
        ind0 = beads.select_bead_from_right(
            amp_bead, num_beads
        )
        ind0 = min(ind0, num_beads-1)
        indf = num_beads
    inds = np.arange(ind0, indf)
    continuous_inds = True

    axis = linalg.uniform_sample_unit_sphere()
    fulcrum = polymer.r[indf-1] if rotate_LHS else polymer.r[ind0]
    r_points = np.ones((4, indf-ind0))
    r_points[0:3] = polymer.r[inds].T
    t3_points = np.zeros((4, indf-ind0))
    t3_points[0:3] = polymer.t3[inds].T
    t2_points = np.zeros((4, indf-ind0))
    t2_points[0:3] = polymer.t2[inds].T

    r_trial, t3_trial, t2_trial = conduct_end_pivot(
        r_points, t3_points, t2_points, axis, fulcrum, rot_angle
    )
    r_trial = r_trial[0:3].T
    t3_trial = t3_trial[0:3].T
    t2_trial = t2_trial[0:3].T

    # print(process_time() - start)

    return (
        inds, r_trial, t3_trial, t2_trial, None, continuous_inds, len(inds),
        rot_angle
    )


def conduct_end_pivot(
    r_points: np.ndarray,
    t3_points: np.ndarray,
    t2_points: np.ndarray,
    axis: np.ndarray,
    fulcrum: np.ndarray,
    rot_angle: float
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
    axis : array_like (3,)
        Rotation axis
    fulcrum : array_like (3,)
        Rotation fulcrum
    rot_angle : float
        Angle of rotation. Positive rotation is counterclockwise when
        rotation axis is pointing directly out of the screen

    Returns
    -------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following rotation
    t3_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    t2_trial : array_like (4, N)
        Homogeneous tangent vectors, orthogonal to t3_trial, following rotation
    """
    rot_matrix = linalg.arbitrary_axis_rotation(
        axis, fulcrum, rot_angle
    )
    r_trial = rot_matrix @ r_points
    t3_trial = rot_matrix @ t3_points
    t2_trial = rot_matrix @ t2_points
    return r_trial, t3_trial, t2_trial


def slide(
    polymer: Polymer,
    amp_move: MOVE_AMP,
    amp_bead: BEAD_AMP
) -> _proposal_arg_types:
    """
    Randomly translate a segment of the polymer.

    Randomly set the translation distance based on the slide move amplitude.
    Then randomly partition this translation distance into x, y, z components.
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
    # print("Slide move")
    # start = process_time()
    translation_amp = amp_move * (np.random.rand())
    rand_z = np.random.rand()
    rand_angle = np.random.rand() * 2 * np.pi
    slide_x = np.sqrt(1 - rand_z**2) * translation_amp * np.cos(rand_angle)
    slide_y = np.sqrt(1 - rand_z**2) * translation_amp * np.sin(rand_angle)
    slide_z = translation_amp * rand_z

    num_beads = polymer.num_beads
    bound_0 = np.random.randint(num_beads)
    bound_1 = max(
        int(beads.select_bead_from_point(amp_bead, num_beads, bound_0)), 1
    )
    ind0, indf = beads.check_bead_bounds(bound_0, bound_1, num_beads)
    inds = np.arange(ind0, indf)
    continuous_inds = True

    r_points = np.ones((4, indf-ind0))
    r_points[0:3] = polymer.r[inds].T
    r_trial = conduct_slide(r_points, slide_x, slide_y, slide_z)
    r_trial = r_trial[0:3].T

    # print(process_time() - start)

    return (
        inds, r_trial, None, None, None, continuous_inds, len(inds),
        translation_amp
    )


def conduct_slide(
    r_points: np.ndarray, x: float, y: float, z: float
) -> np.ndarray:
    """
    Deterministic component of slide move.

    Conduct the slide move on set of beads with homogeneous coordinates stored
    in r_points.

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing slide move
    x : float
        Translation in the x-direction
    y : float
        Translation in the y-direction
    z : float
        Translation in the z-direction

    Return
    ------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following slide move

    """
    translation_mat = linalg.generate_translation_mat(x, y, z)
    r_trial = translation_mat @ r_points
    return r_trial


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
    # print("Tangent Rotation")
    # start = process_time()
    num_beads_to_move = max(int(np.random.uniform() * amp_bead), 1)
    num_beads = polymer.num_beads
    rot_angle = amp_move * (np.random.rand() - 0.5)

    inds = []
    inds_t3_trial = []
    inds_t2_trial = []
    for i in range(num_beads_to_move):
        ind, ind_t3_trial, ind_t2_trial = one_bead_tangent_rotation(
            polymer, inds, rot_angle, num_beads
        )
        inds.append(ind)
        inds_t3_trial.append(ind_t3_trial)
        inds_t2_trial.append(ind_t2_trial)

    inds = np.array(inds)
    inds_t3_trial = np.atleast_2d(np.array(inds_t3_trial))
    inds_t2_trial = np.atleast_2d(np.array(inds_t2_trial))
    t3_trial = inds_t3_trial[:, 0:3]
    t2_trial = inds_t2_trial[:, 0:3]
    continuous_inds = False

    inds, t3_trial, t2_trial = zip(*sorted(zip(inds, t3_trial, t2_trial)))
    t3_trial = np.vstack(t3_trial)
    t2_trial = np.vstack(t2_trial)

    # print(process_time() - start)

    return (
        inds, None, t3_trial, t2_trial, None, continuous_inds, len(inds),
        rot_angle
    )


def one_bead_tangent_rotation(
    polymer: Polymer,
    adjusted_beads: List[int],
    rot_angle: float,
    num_beads: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Random bead rotation for a single bead.

    Stochastic component of random tangent vector rotations. Begins by
    generating a random rotation angle based on the move amplitude. Select a
    random bead which has not yet been rotation by the tangent rotation in this
    MC step. Select an arbitrary axis about which to conduct the rotation. Then
    generate and return trial orientations for the bead undergoing rotation.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    adjusted_beads : List[int]
        List of beads affected by the tangent rotation move
    rot_angle : float
        Random angle by which to rotate a randomly selected bead
    num_beads : int
        Number of beads in the polymer

    Returns
    -------
    ind0 : int
        Index of the rotated bead
    t3_trial : np.ndarray
        Trial t3 tangent of the rotated bead
    t2_trial : np.ndarray
        Trial t2 tangent of the rotated bead
    """
    t3_point = np.zeros(4)
    t2_point = np.zeros(4)
    redraw = True
    while redraw:
        ind0 = np.random.randint(num_beads)
        if ind0 not in adjusted_beads:
            redraw = False

    t3_point[0:3] = polymer.t3[ind0]
    t2_point[0:3] = polymer.t2[ind0]

    axis = linalg.uniform_sample_unit_sphere()

    t3_trial, t2_trial = conduct_tangent_rotation(
        t3_point, t2_point, axis, rot_angle
    )

    return ind0, t3_trial, t2_trial


def conduct_tangent_rotation(
    t3_point: np.ndarray,
    t2_point: np.ndarray,
    axis: np.ndarray,
    rot_angle: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic component of tangent rotation move.

    Rotate tangent vectors `t3_point` and `t2_point` about arbitrary axis for a
    single bead on the polymer.

    Parameters
    ----------
    t3_point : array_like (4,)
        Homogeneous tangent vectors for bead undergoing rotation
    t2_point : array_like (4,)
        Homogeneous tangent, orthogonal to t3 tangent, for rotating bead
    axis : array_like (3,)
        Rotation axis
    rot_angle : float
        Angle of rotation. Positive rotation is counterclockwise when
        rotation axis is pointing directly out of the screen

    Returns
    -------
    t3_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    t2_trial : array_like (4, N)
        Homogeneous tangent vectors, orthogonal to t3 tangents, following
        rotation
    """
    rot_matrix = linalg.arbitrary_axis_rotation(
        axis, np.array([0, 0, 0]), rot_angle
    )
    t3_trial = rot_matrix @ t3_point
    t2_trial = rot_matrix @ t2_point
    return t3_trial, t2_trial


move_list = [crank_shaft, end_pivot, slide, tangent_rotation]


def all_moves(log_dir: str) -> List[MCAdapter]:
    """Generate a list of all adaptable MC move objects.

    NOTE: Use `all_moves` function in `chromo.mc.mc_controller` module to
    create list of controllers for all moves. This function only creates
    list of all moves, which may not be compatible with `mc_sim` function
    in `chromo.mc.mc_sim` module.

    Parameters
    ----------
    log_dir : str
        Path to the directory in which to save log files

    Returns
    -------
    List of all adaptable MC move objects
    """
    return [
        MCAdapter(log_dir + '/' + move.__name__, move) for move in move_list
    ]
