# cython: profile=False

"""Routines for performing Monte Carlo simulations.
"""

import pyximport
pyximport.install()

# Built-in Modules
from typing import List, Optional
from libc.stdlib cimport rand, RAND_MAX

# External Modules
import numpy as np

# Custom Modules
from chromo.polymers import PolymerBase, DetailedChromatinWithSterics
from chromo.polymers cimport PolymerBase, DetailedChromatinWithSterics
from chromo.mc.moves import MCAdapter
from chromo.mc.moves cimport MCAdapter
from chromo.mc.mc_controller import Controller
from chromo.fields cimport UniformDensityField as Udf


cpdef void mc_sim(
    list polymers, readerproteins, long num_mc_steps,
    list mc_move_controllers, Udf field, double mu_adjust_factor,
    long random_seed
):
    """Perform Monte Carlo simulation.

    Pseudocode
    ----------
    Repeat for each Monte Carlo step:
        Repeat for each adaptable move:
            If active, apply move to each polymer
            
    Notes
    -----
    To report process time for each interval of MC steps, add the following:
    
    if (k+1) % 50000 == 0:
        print("MC Step " + str(k+1) + " of " + str(num_mc_steps))
        print(
            "Time for previous 50000 MC Steps (in seconds): ", round(
                process_time()-t1_start, 2
            )
        )
        t1_start = process_time()
        
    TODO: To make the simulator more reproducible, consider setting a random
    seed for the CYTHON code. Note, this uses srand (in libc.stdlib). However,
    be very careful not to accidentally set the random seed in the wrong place
    so as not to disrupt the randomness of the simulation. This is a potential
    source of error.

    Parameters
    ----------
    polymers : List[Polymer]
        Polymers affected by Monte Carlo simulation
    readerproteins : List[ReaderProteins]
        Specification of reader proteins on polymer
    num_mc_steps : int
        Number of Monte Carlo steps to take between save points
    mc_move_controllers : List[Controller]
        List of controllers for active MC moves in simulation
    field: FieldBase (or subclass)
        Field affecting polymer in Monte Carlo simulation
    mu_adjust_factor : double
        Adjustment factor applied to the chemical potential in response to
        simulated annealing
    random_seed : int
        Randoms seed with which to initialize simulation
    """
    cdef bint active_field, update_pairwise_distances
    cdef long i, j, k, n_polymers
    cdef list active_fields
    cdef poly

    np.random.seed(random_seed)
    n_polymers = len(polymers)
    if field.confine_type == "":
        active_fields = [poly.is_field_active() for poly in polymers]
    else:
        active_fields = [1 for _ in polymers]

    for k in range(num_mc_steps):
        for controller in mc_move_controllers:
            if controller.move.move_on == 1:
                for j in range(controller.move.num_per_cycle):
                    for i in range(len(polymers)):
                        poly = polymers[i]
                        # Update distances depending on the class of the polymer
                        update_pairwise_distances = \
                            isinstance(poly, DetailedChromatinWithSterics)
                        poly.mu_adjust_factor = mu_adjust_factor
                        active_field = active_fields[i]
                        mc_step(
                            controller.move, poly, readerproteins, field,
                            active_field, update_pairwise_distances
                        )
            controller.update_amplitudes()


cpdef void mc_step(
    MCAdapter adaptible_move, PolymerBase poly, readerproteins,
    Udf field, bint active_field, bint update_distances
):
    """Compute energy change and determine move acceptance.

    Notes
    -----
    Get the proposed state of the polymer. Calculate the total (polymer +
    field) energy change associated with the move. Accept or reject the move
    based on the Metropolis Criterion.

    After evaluating change in energy (from the polymer and the field), the
    try-except statement checks for RuntimeWarning if the change in energy gets
    too large.

    Parameters
    ----------
    adaptible_move: MCAdapter
        Move applied at particular Monte Carlo step
    poly: PolymerBase
        Polymer affected by move at particular Monte Carlo step
    readerproteins: List[ReaderProteins]
        Reader proteins affecting polymer configuration
    field: FieldBase (or subclass)
        Field affecting polymer in Monte Carlo step
    active_field: bool
        Indicator of whether or not the field is active for the polymer
    update_distances : bint
        Update pairwise distances between beads -- only relevant if the
        polymer is an instance of DetailedChromatinWithSterics, which
        tracks pairwise distances between beads
    """
    cdef double dE, exp_dE
    cdef int check_field = 0
    cdef long packet_size, n_inds
    cdef long[:] inds

    if poly in field and active_field:
        if adaptible_move.name != "tangent_rotation":
            check_field = 1
    packet_size = 20
    inds = adaptible_move.propose(poly)
    n_inds = len(inds)
    if n_inds == 0:
        return
    dE = 0
    dE += poly.compute_dE(adaptible_move.name, inds, n_inds)
    if check_field == 1:
        if adaptible_move.name == "change_binding_state":
            dE += field.compute_dE(
                poly, inds, n_inds, packet_size, state_change=1
            )
        else:
            dE += field.compute_dE(
                poly, inds, n_inds, packet_size, state_change=0
            )
    try:
        exp_dE = np.exp(-dE)
    except RuntimeWarning:
        if dE > 0:
            exp_dE = 0
        elif dE < 0:
            exp_dE = 1

    if (<double>rand() / RAND_MAX) < exp_dE:
        adaptible_move.accept(
            poly, dE, inds, n_inds, log_move=False, log_update=False,
            update_distances=update_distances
        )
        if check_field == 1:
            field.update_affected_densities()
    else:
        adaptible_move.reject(
            poly, dE, inds, n_inds, log_move=False, log_update=False,
            update_distances=update_distances
        )
