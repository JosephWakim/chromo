"""Class and function declarations for fields.pyx

Joseph Wakim
August 4, 2021
"""
cimport chromo.polymers as poly
cimport numpy as np

cdef class FieldBase:
    cdef public list polymers
    cdef public long n_polymers
    cdef public binders
    cdef public str confine_type
    cdef public double confine_length

    cdef double compute_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long packet_size, bint state_change
    )
    cdef void update_affected_densities(self)
    cdef double get_confinement_dE(
        self,
        poly.PolymerBase poly,
        long[:] inds,
        long n_inds,
        int trial
    )

cdef class NullField(FieldBase):
    cdef double compute_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long packet_size, bint state_change
    )
    cdef void update_affected_densities(self)

cdef class UniformDensityField(FieldBase):
    cdef public _field_descriptors
    cdef public double x_width, y_width, z_width
    cdef public double[:] width_xyz
    cdef public long nx, ny, nz, n_sub_bins_x, n_sub_bins_y, n_sub_bins_z
    cdef public double dx, dy, dz
    cdef public double[:] dxyz
    cdef public long n_bins, n_points
    cdef public double vol_bin
    cdef public long[:, ::1] bin_index
    cdef public long[:, ::1] nbr_inds_with_trial
    cdef public long[:] nbr_inds, index_xyz
    cdef public double[:] wt_vec, xyz, weight_xyz
    cdef public double[:, ::1] wt_vec_with_trial, xyz_with_trial
    cdef public double[:, ::1] weight_xyz_with_trial
    cdef public long[:, ::1] index_xyz_with_trial
    cdef public long num_binders
    cdef public long[:] doubly_bound, doubly_bound_trial
    cdef public double[:, ::1] density, density_trial
    cdef public dict access_vols
    cdef public double chi, sub_bin_width_x, sub_bin_width_y, sub_bin_width_z
    cdef public dict sub_bins_to_weights_x, sub_bins_to_weights_y
    cdef public dict sub_bins_to_weights_z, sub_bins_to_bins_x
    cdef public dict sub_bins_to_bins_y, sub_bins_to_bins_z
    cdef public dict dict_
    cdef public float vf_limit
    cdef public bint assume_fully_accessible, fast_field
    cdef public list binder_dict
    cdef public double[:] half_width_xyz
    cdef public double[:] half_step_xyz
    cdef public long[:] n_xyz_m1, affected_bins_last_move
    cdef public long[:, :, ::1] inds_xyz_to_super

    cdef void precompute_ind_xyz_to_super(self)
    cdef void init_fast_field(self, long n_points)
    cpdef dict get_accessible_volumes(
            self, long n_side, bint assume_fully_accessible
    )
    cdef double[:, ::1] get_voxel_coords(self, long[:, ::1] xyz_inds)
    cdef long[:] get_split_voxels(
        self, double[:, ::1] xyz_coords, double buffer_dist
    )
    cdef double[:, ::1] define_voxel_subgrid(self, long n_pt_side)
    cdef double get_frac_accessible(
        self, double[:] coords, double[:, ::1] dxyz_point
    )
    cdef double compute_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long packet_size, bint state_change
    )
    cdef long[:] get_change_in_density(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        bint state_change
    )
    cdef long[:] get_change_in_density_quickly(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        bint state_change
    )
    cdef void _generate_weight_vector_with_trial(self)
    cdef void _generate_index_vector_with_trial(self)
    cdef double get_dE_binders_and_beads(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long[:] bin_inds, bint state_change
    )
    cdef double nonspecific_interact_dE(
        self, poly.PolymerBase poly, long[:] bin_inds, long n_bins
    )
    cdef double[:, ::1] get_volume_fractions_with_trial(
        self, double bead_V, long[:] bin_inds, long n_bins
    )
    cdef void count_doubly_bound(
        self, poly.PolymerBase poly, long[:] inds, long n_inds, bint trial,
        bint state_change
    )
    cpdef double compute_E(self, poly.PolymerBase poly)
    cdef void update_affected_densities(self)
    cpdef void update_all_densities(
        self, poly.PolymerBase poly, long[:]& inds, long n_inds
    )
    cpdef void update_all_densities_for_all_polymers(self)
    cdef void _generate_weight_vector(self)
    cdef void _generate_index_vector(self)
    cdef double get_E_binders_and_beads(
        self, poly.PolymerBase poly, long[:] inds, long n_inds
    )
    cpdef double nonspecific_interact_E(self, poly.PolymerBase poly)
    cdef double[:] get_volume_fractions(self, double bead_V)
    cdef double[:, ::1] get_coordinates_at_inds(
        self,
        double[:, ::1]& r,
        long[:]& inds,
        long& n_inds
    )
    cdef long[:, ::1] get_states_at_inds(
        self,
        poly.PolymerBase poly,
        long[:] inds,
        long n_inds
    )

cdef long inds_to_super_ind(
    long ind_x, long ind_y, long ind_z, long nx, long ny
)
cpdef long[:] super_ind_to_inds(long super_ind, long nx, long ny)
cpdef dict assign_beads_to_bins(
    double[:, ::1] r_poly,
    long n_inds,
    long nx,
    long ny,
    long nz,
    double x_width,
    double y_width,
    double z_width
)
cpdef dict get_neighboring_bins(long nx, long ny, long nz)
cpdef long[:] get_neighbors_at_ind(
    long nx, long ny, long nz, long ind, long num_bins
)
cpdef dict get_blocks(long num_beads, long block_size)
