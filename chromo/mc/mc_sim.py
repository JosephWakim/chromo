"""Routines for performing Monte Carlo simulations.
"""

# Built-in Modules
from time import process_time
from typing import List, TypeVar
import warnings

# External Modules
import matplotlib.pyplot as plt
import numpy as np

# Custom Modules
from chromo.components import Polymer
from chromo.marks import Epigenmark
from chromo.mc.moves import MCAdapter
from chromo.mc.mc_controller import Controller

F = TypeVar("F")    # Represents an arbitrary field


def mc_sim(
    polymers: List[Polymer],
    epigenmarks: List[Epigenmark],
    num_mc_steps: int,
    mc_move_controllers: List[Controller],
    field: F
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
    field: F
        Field affecting polymer in Monte Carlo simulation
    """
    move_runtime = []
    t1_start = process_time()
    for i in range(num_mc_steps):
        if (i+1) % 500 == 0:
            print("MC Step " + str(i+1) + " of " + str(num_mc_steps))
            t1_end = process_time()
            print(
                "Time for previous 500 MC Steps (in seconds): ", round(
                    t1_end-t1_start, 2
                )
            )
            t1_start = process_time()
        for controller in mc_move_controllers:
            if controller.move.move_on:
                for j in range(controller.move.num_per_cycle):
                    for poly in polymers:
                        start = process_time()
                        mc_step(controller.move, poly, epigenmarks, field)
                        controller.update_amplitudes()
                        end = process_time()
                        move_runtime.append(end-start)

    plt.figure(figsize=(4, 3))
    plt.hist(move_runtime, bins=20)
    plt.xlabel("runtime (sec)")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig("doc/source/Runtimes.png")


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
    if poly in field:
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
