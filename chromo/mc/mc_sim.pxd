"""Class and function declarations for mc_sim.pyx

Joseph Wakim
December 9, 2021
"""
from chromo.polymers cimport PolymerBase
from chromo.mc.moves cimport MCAdapter
from chromo.fields cimport UniformDensityField as Udf

cpdef void mc_sim(
    list polymers, readerproteins, long num_mc_steps,
    list mc_move_controllers, Udf field, double mu_adjust_factor,
    long random_seed
)

cpdef void mc_step(
    MCAdapter adaptible_move, PolymerBase poly, readerproteins,
    Udf field, bint active_field, bint update_distances
)