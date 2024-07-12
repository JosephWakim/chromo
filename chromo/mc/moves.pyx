# cython: profile=False

"""Utilities for proposing Monte Carlo moves.

This module includes an MC adapter class which will drive move adaption and
functions applying transformations made by each move.
"""

# Built-in Modules
from typing import Tuple, List, Callable, Optional, Union, Dict

# External Modules
import numpy as np
import pandas as pd

# Custom Modules
import chromo.util.mc_stat as mc_stat
from chromo.mc.move_funcs import (
    change_binding_state, crank_shaft, end_pivot, slide, tangent_rotation
)
from chromo.polymers import PolymerBase
from chromo.polymers cimport PolymerBase
from chromo.binders import ReaderProtein


MOVE_AMP = float
BEAD_AMP = int
NUMERIC = Union[int, float]

_proposal_arg_names = [
    'inds', 'r', 't3', 't2', 'states', 'continuous_inds', 'bead_amp',
    'move_amp'
]

_proposal_arg_types = Tuple[
    Tuple[int, int],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
    List[ReaderProtein],
    bool,
    BEAD_AMP,
    MOVE_AMP
]

_move_arg_types = [
    PolymerBase,
    MOVE_AMP,
    BEAD_AMP
]

_move_func_type = Callable[
    [PolymerBase, MOVE_AMP, BEAD_AMP], _proposal_arg_types
]

# ctypedef long[:] (*mv_fxn)(PolymerBase, double, double)

cdef class MCAdapter:
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

    Success rate will be tracked by a `AcceptanceTracker` object specified in
    the `chromo/util/mc_stat.py` module. The performance tracker will indicate
    an overall acceptance rate based on the total numbers of moves attempted
    and accepted. A running acceptance rate will also be maintained, which
    decays the weight of historic values using a decay factor.

    In the future, we want to do this optimization by looking at actual metrics
    based on the simulation output, but for now only implementation of an
    MCAdapter (`MCAdaptSuccessRate`) handles this optimization by simply trying
    to maintain a fixed target success rate.
    """
    def __init__(
        self,
        str log_dir,
        str log_file_prefix,
        move_func,  # define as type `mv_fxn` if only accessed from C
        float moves_in_average,
        long init_amp_bead,
        double init_amp_move,
    ):
        """Initialize the MCAdapter object.

        Parameters
        ----------
        log_dir : str
            Path to the directory into which to save logs of bead/move
            amplitudes and acceptance rates
        log_file_prefix : str
            File prefix for log file tracking bead/move amplitudes and
            acceptance rate
        move_func : Callable[_move_arg_types, _proposal_arg_types]
            Functions representing Monte Carlo moves
        moves_in_average : Optional[float]
            Number of historical moves to track in incremental measure of move
            acceptance (default = 20)
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
        self.move_on = 1
        self.last_amp_move = 0
        self.last_amp_bead = 0
        self.acceptance_tracker = mc_stat.AcceptanceTracker(
            log_dir, log_file_prefix, moves_in_average
        )

    def __str__(self):
        return f"MCAdapter<{self.name}>"

    def to_file(self, path):
        pass

    cdef long[:] propose(self, PolymerBase polymer):
        """
        Get new proposed state of the system.

        Parameters
        ----------
        polymer : PolymerBase
            The polymer to attempt to move.

        Returns
        -------
        inds : array_like (N,)
            List of indices for N beads being moved
        """
        self.num_attempt += 1
        return self.move_func(
            polymer=polymer, amp_move=self.amp_move, amp_bead=self.amp_bead
        )

    cpdef void accept(
        self, PolymerBase poly, double dE, long[:] inds, long n_inds,
        bint log_move, bint log_update, bint update_distances
    ):
        """Update polymer with new state and update proposal stats.

        Update all elements of `poly` for which proposed state is not None. Log
        the move acceptance/rejection in the move acceptance tracker.

        `replace_none` is outdated with cython implementation of `move_funcs`

        Parameters
        ----------
        poly : PolymerBase
            Polymer affected by the MC move
        dE : float
            Change in energy associated with the move
        inds : long[:]
            Indices of beads affected by the MC move
        n_inds : long
            Number of beads affected by a move
        log_move : bint
            Indicator for whether (1) or not (0) to log the move to a list
            later outputted to a CSV file
        log_update : bint
            Indicator for whether (1) or not (2) to record the updated
            acceptance rate after the MC move
        update_distances : bint
            Update pairwise distances between beads -- only relevant if the
            polymer is an instance of DetailedChromatinWithSterics, which
            tracks pairwise distances between beads.
        """
        cdef long i, j

        if self.name == "change_binding_state":
            for i in range(n_inds):
                for j in range(poly.num_binders):
                    poly.states[inds[i], j] = poly.states_trial[inds[i], j]

        elif self.name == "slide":
            for i in range(n_inds):
                for j in range(3):
                    poly.r[inds[i], j] = poly.r_trial[inds[i], j]
                    poly.t3_trial[inds[i], j] = poly.t3[inds[i], j]
                    poly.t2_trial[inds[i], j] = poly.t2[inds[i], j]
                if update_distances:
                    for j in range(poly.num_beads):
                        poly.distances[inds[i], j] = \
                            poly.distances_trial[inds[i], j]
                        poly.distances[j, inds[i]] = \
                            poly.distances_trial[j, inds[i]]

        elif self.name == "tangent_rotation":
            for i in range(n_inds):
                for j in range(3):
                    poly.t3[inds[i], j] = poly.t3_trial[inds[i], j]
                    poly.t2[inds[i], j] = poly.t2_trial[inds[i], j]
                    poly.r_trial[inds[i], j] = poly.r[inds[i], j]

        else:
            for i in range(n_inds):
                for j in range(3):
                    poly.r[inds[i], j] = poly.r_trial[inds[i], j]
                    poly.t3[inds[i], j] = poly.t3_trial[inds[i], j]
                    poly.t2[inds[i], j] = poly.t2_trial[inds[i], j]
                if update_distances:
                    for j in range(poly.num_beads):
                        poly.distances[inds[i], j] = \
                            poly.distances_trial[inds[i], j]
                        poly.distances[j, inds[i]] = \
                            poly.distances_trial[j, inds[i]]

        self.num_success += 1
        self.acceptance_tracker.update_acceptance_rate(
            accept=1.0, log_update=log_update
        )
        if log_move == 1:
            self.acceptance_tracker.log_move(
                self.amp_move,
                self.amp_bead,
                poly.last_amp_move,
                poly.last_amp_bead,
                dE
            )

    cpdef void reject(
        self, PolymerBase poly, double dE, long[:] inds, long n_inds,
        bint log_move, bint log_update, bint update_distances
    ) except *:
        """Reject a proposed Monte Carlo move.

        Log the rejected move in the acceptance tracker.

        Parameters
        ----------
        poly : PolymerBase
            Polymer affected by the MC move
        dE : float
            Change in energy associated with the move
        inds : long[:]
            Indices of the beads that were affected by the move
        n_inds : long
            Number of beads that were affected by the move
        log_move : bint
            Indicator for whether (1) or not (0) to log the move to a list
            later outputted to a CSV file
        log_update : bint
            Indicator for whether (1) or not (2) to record the updated
            acceptance rate after the MC move
        update_distances : bint
            Update pairwise distances between beads -- only relevant if the
            polymer is an instance of DetailedChromatinWithSterics, which
            tracks pairwise distances between beads.
        """
        # Reset trial states
        if self.name == "change_binding_state":
            for i in range(n_inds):
                for j in range(poly.num_binders):
                    poly.states_trial[inds[i], j] = poly.states[inds[i], j]
        else:
            for i in range(n_inds):
                for j in range(3):
                    poly.r_trial[inds[i], j] = poly.r[inds[i], j]
                    poly.t3_trial[inds[i], j] = poly.t3[inds[i], j]
                    poly.t2_trial[inds[i], j] = poly.t2[inds[i], j]
                if update_distances and (
                    self.name == "slide" or self.name == "end_pivot" or
                    self.name == "crank_shaft"
                ):
                    for j in range(poly.num_beads):
                        poly.distances_trial[inds[i], j] = \
                            poly.distances[inds[i], j]
                        poly.distances_trial[j, inds[i]] = \
                            poly.distances[j, inds[i]]

        # Update acceptance tracker
        self.acceptance_tracker.update_acceptance_rate(
            accept=0.0, log_update=log_update
        )
        if log_move == 1:
            self.acceptance_tracker.log_move(
                self.amp_move, self.amp_bead, poly.last_amp_move,
                poly.last_amp_bead, dE
            )


cdef class Bounds:
    """Class representation of move or bead amplitude bounds.
    """

    def __init__(self, str name, dict bounds):
        """Initialize the `Bounds` object.

        Parameters
        ----------
        name : str
            Name of the bounds
        bounds : Dict[str, Tuple[NUMERIC, NUMERIC]]
            Dictionary of bead selection or move amplitude bounds for each move
            type, where keys are the names of the move types and values are
            tuples in the form (lower bound, upper bound)
        """
        self.name = name
        self.bounds = bounds

    def to_dataframe(self):
        """Express the Bounds using a dataframe.
        """
        move_names = self.bounds.keys()
        bounds_arr = np.atleast_2d(
            np.array(list(self.bounds.values())).flatten()
        )
        column_names = pd.MultiIndex.from_product(
            [move_names, ('lower_bound', 'upper_bound')]
        )
        df = pd.DataFrame(bounds_arr, columns=column_names)
        return df

    def to_csv(self, path):
        """Save Polymer object to CSV file as DataFrame.
        """
        return self.to_dataframe().to_csv(path)

    def to_file(self, path):
        """Synonym for `to_csv` to conform to `make_reproducible` spec.
        """
        return self.to_csv(path)


cpdef list move_list = [
    crank_shaft, end_pivot, slide, tangent_rotation, change_binding_state
]


cpdef list all_moves(str log_dir):
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
