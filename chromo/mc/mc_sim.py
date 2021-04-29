"""Routines for performing Monte Carlo simulations.
"""

# Built-in Modules
from typing import List
import warnings

# External Modules
import numpy as np

# Custom Modules
from chromo.components import Polymer
from chromo.marks import Epigenmark
from chromo.fields import FieldBase
from chromo.mc.moves import MCAdapter
from chromo.mc.mc_controller import Controller


def mc_sim(
    polymers: List[Polymer],
    epigenmarks: List[Epigenmark],
    num_mc_steps: int,
    mc_move_controllers: List[Controller],
    field: FieldBase
):
    """Perform Monte Carlo simulation.

    Repeat for each Monte Carlo step:
        Repeat for each adaptable move:
            If active, apply move to each polymer

    Parameters
    ----------
    polymers: List[Polymer]
        Polymers affected by Monte Carlo simulation
    epigenmarks: List[Epigenmark]
        Specification of epigenetic marks on polymer
    num_mc_steps: int
        Number of Monte Carlo steps to take between save points
    mc_move_controllers: List[Controller]
        List of controllers for active MC moves in simulation
    field: FieldBase
        Field affecting polymer in Monte Carlo simulation
    """
    for i in range(num_mc_steps):

        if (i+1) % 500 == 0:
            print("MC Step " + str(i+1) + " of " + str(num_mc_steps))

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
    field: FieldBase
):
    """Compute energy change and determine move acceptance.

    Get the proposed state of the polymer. Calculate the total (polymer +
    field) energy change associated with the move. Accept or reject the move
    based on the Metropolis Criterion.

    Parameters
    ----------
    adaptible_move: MCAdapter
        Move applied at particular Monte Carlo step
    poly: Polymer
        Polymer affected by move at particular Monte Carlo step
    epigenmarks: List[Epigenmark]
        Epigenetic marks affecting polymer configuration
    field: FieldBase
        Field affecting polymer in Monte Carlo step
    """
    proposal = adaptible_move.propose(poly)
    adaptible_move.last_amp_bead = proposal[-2]
    adaptible_move.last_amp_move = proposal[-1]

    dE = 0
    dE += poly.compute_dE(*proposal[:-2])
    if poly in field:
        dE += field.compute_dE(poly, *proposal[:-2])

    # Catch RuntimeWarning if the change in energy gets too large
    warnings.filterwarnings("error")
    try:
        exp_dE = np.exp(-dE)
    except RuntimeWarning:
        exp_dE = 0

    if np.random.rand() < exp_dE:       # accept
        adaptible_move.accept(poly, *proposal)
    else:                               # reject
        adaptible_move.reject()
