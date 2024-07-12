"""Class and function declarations for moves.pyx

Joseph Wakim
December 9, 2021
"""

from chromo.polymers cimport PolymerBase
from chromo.mc.move_funcs import (
    change_binding_state, crank_shaft, end_pivot, slide, tangent_rotation
)

# ctypedef long[:] (*mv_fxn)(PolymerBase, double, double)

cdef class MCAdapter:
    cdef public str name
    cdef public move_func   # define as type `mv_fxn` if only accessed from C
    cdef public double amp_move
    cdef public long num_per_cycle
    cdef public long amp_bead
    cdef public long num_attempt
    cdef public long num_success
    cdef public bint move_on
    cdef public double last_amp_move
    cdef public long last_amp_bead
    cdef public acceptance_tracker

    cdef long[:] propose(self, PolymerBase polymer)
    cpdef void accept(
        self, PolymerBase poly, double dE, long[:] inds, long n_inds,
        bint log_move, bint log_update, bint update_distances
    )
    cpdef void reject(
        self, PolymerBase poly, double dE,  long[:] inds, long n_inds,
        bint log_move, bint log_update, bint update_distances
    ) except *

cdef class Bounds:
    cdef public str name
    cdef public dict bounds

cpdef list move_list = [
    crank_shaft, end_pivot, slide, tangent_rotation, change_binding_state
]

cpdef list all_moves(str log_dir)
