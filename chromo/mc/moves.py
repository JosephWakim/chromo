"""Utilities for proposing Monte Carlo moves.

This module include an MC adapter class which will drive move adaption and
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
    crank_shaft, end_pivot, slide, tangent_rotation
)
from chromo.polymers import Polymer
from chromo.marks import Epigenmark


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
        log_path: str,
        move_func: _move_func_type,
        moves_in_average: Optional[float] = 20,
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
        self.move_on = True
        self.last_amp_move = None
        self.last_amp_bead = None

        self.acceptance_tracker =\
            mc_stat.AcceptanceTracker(log_path, moves_in_average)

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
        continuous_inds = proposal[-3]

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
                if continuous_inds:
                    prop = poly.__dict__[name][inds]
                else:
                    prop = np.array([poly.__dict__[name][ind] for ind in inds])
                actual_proposal.append(prop)

        return actual_proposal

    def accept(self, poly, dE, *proposal):
        """Update polymer with new state and update proposal stats.

        Update all elements of `poly` for which proposed state is not None. Log
        the move acceptance/rejection in the move acceptance tracker.

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
        continuous_inds = proposal[-3]
        if continuous_inds:
            poly.r[inds] = r
            poly.t3[inds] = t3
            poly.t2[inds] = t2
            poly.states[inds] = states
        else:
            for i in range(len(inds)):
                poly.r[inds[i]] = r[i]
                poly.t3[inds[i]] = t3[i]
                poly.t2[inds[i]] = t2[i]
                poly.states[inds[i]] = states[i]
        self.num_success += 1
        self.acceptance_tracker.update_acceptance_rate(accept=True)
        self.acceptance_tracker.log_move(
            self.amp_move,
            self.amp_bead,
            self.last_amp_move,
            self.last_amp_bead,
            dE
        )

    def reject(self, dE):
        """Reject a proposed Monte Carlo move.

        Log the rejected move in the acceptance tracker.

        Parameters
        ----------
        dE : float
            Change in energy associated with the move
        """
        self.acceptance_tracker.update_acceptance_rate(accept=False)
        self.acceptance_tracker.log_move(
            self.amp_move,
            self.amp_bead,
            self.last_amp_move,
            self.last_amp_bead,
            dE
        )


class Bounds(object):
    """Class representation of move or bead amplitude bounds.
    """

    def __init__(self, name: str, bounds: Dict[str, Tuple[NUMERIC, NUMERIC]]):
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
