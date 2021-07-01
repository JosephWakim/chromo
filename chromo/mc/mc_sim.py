"""Routines for performing Monte Carlo simulations.
"""

# Built-in Modules
from time import process_time
from typing import List, TypeVar, Optional
import warnings

# External Modules
import numpy as np

# Custom Modules
from chromo.polymers import Polymer
from chromo.marks import Epigenmark
from chromo.mc.moves import MCAdapter
from chromo.mc.mc_controller import Controller

F = TypeVar("F")    # Represents an arbitrary field


def mc_sim(
    polymers: List[Polymer],
    epigenmarks: List[Epigenmark],
    num_mc_steps: int,
    mc_move_controllers: List[Controller],
    field: F,
    random_seed: Optional[int] = 0
):
    """Perform Monte Carlo simulation.

    Repeat for each Monte Carlo step:
        Repeat for each adaptable move:
            If active, apply move to each polymer

    Parameters
    ----------
    polymers : List[Polymer]
        Polymers affected by Monte Carlo simulation
    epigenmarks : List[Epigenmark]
        Specification of epigenetic marks on polymer
    num_mc_steps : int
        Number of Monte Carlo steps to take between save points
    mc_move_controllers : List[Controller]
        List of controllers for active MC moves in simulation
    field: F
        Field affecting polymer in Monte Carlo simulation
    random_seed : Optional[int]
        Randoms seed with which to initialize simulation (default = 0)
    """
    np.random.seed(random_seed)
    t1_start = process_time()

    for i in range(num_mc_steps):
        if (i+1) % 500 == 0:
            print("MC Step " + str(i+1) + " of " + str(num_mc_steps))
            print(
                "Time for previous 500 MC Steps (in seconds): ", round(
                    process_time()-t1_start, 2
                )
            )
            t1_start = process_time()

        for controller in mc_move_controllers:
            if controller.move.move_on:
                for j in range(controller.move.num_per_cycle):
                    for poly in polymers:
                        mc_step(controller.move, poly, epigenmarks, field)
                        controller.update_amplitudes()


def mc_step(
    adaptible_move: MCAdapter,
    poly: Polymer,
    epigenmarks: List[Epigenmark],
    field: F
):
    """Compute energy change and determine move acceptance.

    Get the proposed state of the polymer. Calculate the total (polymer +
    field) energy change associated with the move. Accept or reject the move
    based on the Metropolis Criterion.

    After evaluating change in energy (from the polymer and the field), the
    try-except statement checks for RuntimeWarning if the change in energy gets
    too large.

    NOTE: For now, do not consider that some moves do not affect certain parts
    of the polymer. Instead, fill any `None` values with the old states of the
    polymer explicitly. This may be adjusted during future optimization.

    TODO: Move `poly.is_field_active()` to the outer loop.

    Parameters
    ----------
    adaptible_move: MCAdapter
        Move applied at particular Monte Carlo step
    poly: Polymer
        Polymer affected by move at particular Monte Carlo step
    epigenmarks: List[Epigenmark]
        Epigenetic marks affecting polymer configuration
    field: F
        Field affecting polymer in Monte Carlo step
    """
    proposal = adaptible_move.propose(poly)
    proposal = list(proposal)
    adaptible_move.last_amp_bead = proposal[-2]
    adaptible_move.last_amp_move = proposal[-1]

    proposal[1:5] = MCAdapter.replace_none(poly, *proposal)
    for i in range(1, 4):
        proposal[i] = np.array(proposal[i])

    dE = 0
    dE += poly.compute_dE(*proposal[:-2])
    if poly in field and poly.is_field_active():
        dE += field.compute_dE(poly, *proposal[:-2])

    warnings.filterwarnings("error")
    try:
        exp_dE = np.exp(-dE)
    except RuntimeWarning:
        exp_dE = 0

    if np.random.rand() < exp_dE:
        adaptible_move.accept(poly, dE[0], *proposal)
    else:
        adaptible_move.reject(dE[0])
