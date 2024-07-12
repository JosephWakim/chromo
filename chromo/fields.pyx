# cython: profile=True
"""Fields discretize space to efficiently calculate change in binder energy.

Notes
-----
Creates a field object that contains parameters for the field calculations
and functions to generate the densities.

Cythonized to accomodate expensive `compute_dE` method.

To profile runtime of cython functions, add the following comment to the top of
this file: # cython: profile=True
"""

import pyximport
pyximport.install()

# Built-in Modules
from pathlib import Path
from libc.math cimport floor, sqrt

# External Modules
import numpy as np
cimport numpy as np
import pandas as pd

# Custom Modules
import chromo.polymers as poly
from chromo.util.linalg cimport vec_dot3


cdef double E_HUGE = 1E99

cdef list _field_descriptors = [
    'x_width', 'nx', 'y_width', 'ny', 'z_width', 'nz', 'confine_type',
    'confine_length', 'chi', 'assume_fully_accessible', 'vf_limit', 'fast_field'
]
cdef list _int_field_descriptors = ['nx', 'ny', 'nz', 'n_points']
cdef list _str_field_descriptors = ['confine_type']
cdef list _float_field_descriptors = [
    'x_width', 'y_width', 'z_width', 'confine_length', 'vf_limit', 'chi'
]
cdef list _bool_field_descriptors = ['assume_fully_accessible', 'fast_field']


cdef class FieldBase:
    """A discretization of space for computing energies.

    Notes
    -----
    Must be subclassed to be useful.

    As implemented so far, this codebase accepts only one polymer in the field

    Attributes
    ----------
    polymers : List[PolymerBase]
        Polymers contained in the field; for now, this codebase accepts only
        one polymer in the field
    n_polymers : long
        Number of polymers in the field
    binders : pd.DataFrame
        Table containing reader proteins bound to the polymer and their
        relevant properties
    """

    @property
    def name(self):
        """Print the name of the field.

        Notes
        -----
        For now, there's only one field per sim, so classname works.
        """
        return self.__class__.__name__

    def __init__(self, polymers, binders):
        """Construct a field holding no polymers, tracking no reader proteins.

        Parameters
        ----------
         polymers : List[PolymerBase]
            List of polymers contained in the field
        binders : pd.DataFrame
            Output of `chromo.binders.make_binder_collection` applied to the
            list of `Binder` objects contained in the field
        """
        self.polymers = polymers
        self.n_polymers = len(polymers)
        self.binders = binders
        self.confine_type = ""
        self.confine_length = 0.0

    def __str__(self):
        """Print representation of empty field.
        """
        return "Field<>"

    def __contains__(self, poly):
        """Check if a polymer is currently set to interact with this field.

        Notes
        -----
        No polymers interact with the base field object.
        """
        return self.polymers.__contains__(poly)

    cdef double compute_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long packet_size, bint state_change
    ):
        """Compute the change in field energy due to a proposed move.
        """
        raise NotImplementedError, "Reading compute_dE from base class"

    cdef void update_affected_densities(self):
        raise NotImplementedError, \
            "Reading update_affected_densities from base class"
        pass

    cdef double get_confinement_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_beads, int trial
    ):
        """Evaluate the energy associated with a confining boundary.

        Notes
        -----
        A large energy is assigned to any point lies outside the confinement;
        otherwise, an energy contribution of zero is returned.

        This method is where different confinement types can be defined.
        However, if a non-spherical confinement is defined, adjustments need
        to be made to `self.get_accessible_volumes()`, which currently supports
        spherical confinements.

        Parameters
        ----------
        poly : poly.PolymerBase
            The polymer which has been moved.
        inds : long[:]
            Vector of bead indices for which to evaluate confinement
        n_beads : long
            Number of beads for which to evaluate confinement
        trial : int
            Indicator of whether to evaluate confinement on trial positions (1)
            or current positions (0)

        Returns
        -------
        double
            Energy contribution of the confinement; either zero if all points
            lie inside the confinement or some large value ensuring the move
            gets rejected if any point lies outside the confinement
        """
        cdef long i
        cdef long num_out_of_bounds = 0
        cdef double dist

        # No Confinement
        if self.confine_type == "":
            return 0.

        # Spherical confinement
        elif self.confine_type == "Spherical":
            if trial == 1:
                for i in range(n_beads):
                    dist = sqrt(
                        vec_dot3(poly.r_trial[inds[i]], poly.r_trial[inds[i]])
                    )
                    if dist > self.confine_length:
                        num_out_of_bounds += 1
            elif trial == 0:
                for i in range(n_beads):
                    dist = sqrt(vec_dot3(poly.r[inds[i]], poly.r[inds[i]]))
                    if dist > self.confine_length:
                        num_out_of_bounds += 1
            else:
                raise ValueError("Trial flag " + str(trial) + " not found.")
            return num_out_of_bounds * E_HUGE

        # Cubical confinement
        elif self.confine_type == "Cubical":
            if trial == 1:
                for i in range(n_beads):
                    for j in range(3):
                        if np.abs(poly.r_trial[inds[i], j]) > self.confine_length / 2:
                            num_out_of_bounds += 1
            elif trial == 0:
                if trial == 1:
                    for i in range(n_beads):
                        for j in range(3):
                            if np.abs(poly.r[inds[i], j]) > self.confine_length / 2:
                                num_out_of_bounds += 1
            else:
                raise ValueError("Trial flag " + str(trial) + " not found.")
            return num_out_of_bounds * E_HUGE

        # Confinement type not found
        else:
            raise ValueError(
                "Confinement type " + self.confine_type + " not found."
            )


class Reconstructor:
    """Defer defining `Field` until after `PolymerBase`/`Binder` instances.

    Notes
    -----
    Constructs a kwargs object that can be re-passed to the appropriate `Field`
    constructor when they become available.

    Parameters
    ----------
    field_constructor : cls
        Class from which the field will be instantiated
    kwargs : Dict
        Keyword arguments used to instantiate the field
    """

    def __init__(self, cls, **kwargs):
        """Construct our Reconstructor.

        Parameters
        ----------
        field_constructor : cls
            Class from which the field will be instantiated
        """
        self.field_constructor = cls
        self.kwargs = kwargs

    def finalize(self, polymers, binders) -> FieldBase:
        """Finish construction of appropriate `Field` object.

        Parameters
        ----------
        polymers : List[PolymerBase]
            List of polymers contained in the field
        binders : pd.DataFrame
            Table representing reader proteins bound to polymers in the field

        Returns
        -------
        FieldBase
            Field representation of discretized space containing polymers
        """
        return self.field_constructor(
            polymers=polymers, binders=binders, **self.kwargs
        )

    @classmethod
    def from_file(cls, path: Path) -> FieldBase:
        """Assume class name is encoded in file name.

        Parameters
        ----------
        path : Path
            File path object directed to the file defining the field

        Returns
        -------
        FieldBase
            Field representation of discretized space containing polymers
        """
        constructor = globals()[path.name]
        kwargs = pd.read_csv(path).iloc[0].to_dict()
        return cls(constructor, **kwargs)

    def __call__(self, polymers, binders) -> FieldBase:
        """Synonym for `Reconstructor.finalize()`.

        Notes
        -----
        See documentation for `Reconstructor.finalize()` for additional
        details and parameter/returns definitions.
        """
        return self.finalize(polymers, binders)


cdef class NullField(FieldBase):
    """A field with no energy contributions.
    """

    def __init__(self, polymers = None, confine_type="", confine_length=0.0):
        if polymers is None:
            polymers = []
        super(NullField, self).__init__(
            polymers, binders = pd.DataFrame()
        )
        self.confine_type = confine_type
        self.confine_length = confine_length

    def to_file(self, path):
        """Save Field description.
        """
        with open(path, 'w'):
            pass

    @classmethod
    def from_file(cls, path):
        """Load Field description.
        """
        return cls()

    cdef double compute_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long packet_size, bint state_change
    ):
        """Compute the change in energy associated with the null field.
        
        Notes
        -----
        The field energy change associated with the null field can only come
        from the confinement.        
        """
        dE = self.get_confinement_dE(poly, inds, n_inds, trial=1)
        dE -= self.get_confinement_dE(poly, inds, n_inds, trial=0)
        return dE

    cdef void update_affected_densities(self):
        pass


cdef class UniformDensityField(FieldBase):
    """Rectilinear discretization of space as a rectangular box.

    Notes
    -----
    Each bead contributes "mass" to the eight nearest voxels containing it.
    The mass is linearly interpolated based on the distance of the bead from
    the center of each voxel. This is much more stable numerically than just
    using the voxels as "bins," as would be done in e.g., finite-difference
    discretization.

    Many attributes are updated during each calculation of field energy
    change. We choose to track and update these attributes to avoid costly
    array initialization during each iteration of the MC simulation.
    Affected attributes are identified with the "TEMP" flag in their
    description.

    Attributes
    ----------
    _field_descriptors : List[str]
        Names of attributes which describe the geometric layout of the field
        as a grid of voxels
    x_width, y_width, z_width : double
        Dimensions of the discretized field in the x, y, and z directions
    width_xyz : double[:]
        Vectorized representation of the x, y, and z widths of the
        discretized field
    nx, ny, nz : long
        Number of voxels in each of the x, y, and z directions
    dx, dy, dz : double
        Width of each voxel in the x, y, and z directions
    dxyz : double[:]
        Vectorized representation of the voxel widths in the x, y, and z
        directions
    n_bins : long
        Number of voxels in the field
    vol_bin : double
        Volume of each voxel; equal to `dx*dy*dz`
    bin_index : long[:, ::1]
        For each of the `nx*ny*nz` voxels in the field, stores indices
        identifying eight vertices at the corners of the voxel
    nbr_indices_with_trial : long[:, ::1]
        Super-indices of eight neighboring voxels containing a bead's
        position in space (dim1) from the polymer's current configuration
        (dim1=0) and proposed configuration (dim1=1); TEMP
    nbr_inds : long[:]
        Super-indices of eight neighboring voxels containing a bead's
        position in space; TEMP
    index_xyz_with_trial : long[:, ::1]
        Vector containing a bead's bin index in the x, y, and z directions
        (dim1) for the polymer's current configuration (dim0=0) and trial
        configuration (dim0=1); TEMP
    index_xyz : long[:]
        Vector containing a bead's bin index in the x, y, z directions; TEMP
    wt_vec_with_trial : double [:]
        Bead's weight linearly interpolated between the eight nearest voxels
        surrounding it (dim1) for a polymer's current configuration (dim0=0)
        and trial configuration (dim0=1); TEMP
    wt_vec : double[:]
        Vector containing a bead's weight linearly interpolated between the
        eight nearest voxels surrounding it; TEMP
    xyz_with_trial : double[:, ::1]
        Position of the bead in the x, y, z directions (dim1), shifted during
        linear interpolation of voxel weights for the polymer's current
        configuration (dim0=0) and trial configuration (dim0=1); TEMP
    xyz : double[:]
        Position of the bead in the x, y, z directions, shifted during linear
        interpolation of voxel weights; TEMP
    weight_xyz_with_trial : double[:, ::1]
        Vector of a bead's linearly interpolated weights in a voxel in the x,
        y, z directions as determined for the voxel surrounding the bead with
        the lowest x, y, z indices (dim1) for the polymer's current
        configuration (dim0=0) and tiral configuration (dim0=1); TEMP
    weight_xyz : double[:]
        Vector of a bead's linearly interpolated weights in a voxel in the x,
        y, z directions as determined for the voxel surrounding the bead with
        the lowest x, y, z indices; TEMP
    num_binders : long
        Number of reader proteins bound to the polymer in the simulation
    doubly_bound, doubly_bound_trial : long[:]
        Vectors indicating whether or not a bead is doubly bound by each
        tracked reader protein in the current polymer configuration
        (`doubly_bond`) and trial configuration (`doubly_bound_trial`); TEMP
    confine_type : str
        Name of the confining boundary around the polymer; if the polymer is
        unconfined, `confine_type` is a blank string
    confine_length : double
        Length scale of the confining boundary
    density, density_trial : double[:, ::1]
        Current (`density`) and proposed change (`density_trial`) in density
        of beads (column 0) and each reader protein (columns 1...) in each
        voxel of the discretized space; voxels are sorted by super-index and
        arranged down the rows the the density arrays
    access_vols : Dict[long, double]
        Mapping of voxel super-index (keys) to volume of the voxel inside the
        confining boundary (values).
    chi : double
        Negative local Flory-Huggins parameter dictating non-specific bead
        interaction, in units of (kT / bead vol. nondim. by bin vol.)
    dict_ : dict
        Dictionary of key attributes defining the field and their values
    binder_dict : dict
        Dictionary representation of each reader protein and their properties
    half_width_xyz : double[:]
        Half the width of the full discretized space in the x, y,
        and z directions; equivalent to `np.array([self.x_width/2,
        self.y_width/2, self.z_width/2])`
    half_step_xyz : double[:]
        Half the length of the voxel in single x, y, and z directions;
        equivalent to `np.array([self.dx/2, self.dy/2, self.dz/2])`
    n_xyz_m1 : long[:]
        One less than the number of voxels that fit in the x, y, and z
        directions of the  field.
    inds_xyz_to_super : long[:, :, ::1]
        Array containing the super-index of each voxel (stored in the array
        values) for each combination of x-voxel position (dim0), y-voxel
        position (dim1) and z-voxel position (dim2).
    vf_limit : float
            Volume fraction limit in a voxel.
    fast_field : bint
        If this value is `1`, a coarse-grained density field will be
        evaluated using pre-computed sub-bin weightings; otherwise, bin
        weightings will be interpolated from bead positions during each
        iteration
    n_points : long
        Number of sub-bins to precompute in each dimension when the ``fast
        field'' is active; should be an even number
    """

    def __init__(
        self, polymers, binders, x_width, nx, y_width, ny, z_width, nz,
        confine_type = "", confine_length = 0.0, chi = 1.0,
        assume_fully_accessible = 1, vf_limit = 0.5, fast_field = 0,
        n_points = 1000
    ):
        """Construct a `UniformDensityField` containing polymers.

        Parameters
        ----------
        polymers : List[PolymerBase]
            List of polymers contained in the field
        binders : pd.DataFrame
            Output of `chromo.binders.make_binder_collection` applied to the
            list of `Binder` objects contained in the field
        x_width, y_width, z_width : double
            Width of the box containing the field in the x, y, and z-directions
        nx, ny, nz : long
            Number of bins in the x, y, and z-directions
        confine_type : str
            Name of the confining boundary; to indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary; length
            representation specified in function associated w/ `confine_type`
        chi : double
            Negative local Flory-Huggins parameter dictating non-specific bead
            interaction (default = 1)
        assume_fully_accessible : bint
            Flag indicating whether to assume all voxels are fully accessible.
            Assume voxels are fully accessible if the voxel volumes are far less
            than the confinement volume. Value of `1` indicates that all voxels
            are assumed to be fully accessible, bypassing the calculation of
            accessible volume (default = 1).
        vf_limit : Optional[float]
            Volume fraction limit in a voxel (default = 0.5)
        fast_field : Optional[bint]
            If this value is `1`, a coarse-grained density field will be
            evaluated using pre-computed sub-bin weightings; otherwise, bin
            weightings will be interpolated from bead positions during each
            iteration (default = 0)
        n_points : Optional[long]
            Number of sub-bins to precompute in each dimension when the ``fast
            field'' is active; should be an even number (default = 1000)
        """
        super(UniformDensityField, self).__init__(
            polymers = polymers, binders = binders
        )
        self._field_descriptors = _field_descriptors
        for poly in polymers:
            if poly.num_binders != len(binders):
                raise NotImplementedError(
                    "For now, all polymers must use all of the same binders."
                )
        self.x_width = x_width
        self.y_width = y_width
        self.z_width = z_width
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.init_grid()
        self.num_binders = len(binders)
        self.doubly_bound = np.zeros((self.num_binders,), dtype=int)
        self.doubly_bound_trial = np.zeros((self.num_binders,), dtype=int)
        self.init_field_energy_prefactors()
        self.density = np.zeros((self.n_bins, self.num_binders+1), dtype=float)
        self.density_trial = self.density.copy()
        self.confine_type = confine_type
        self.confine_length = confine_length
        self.assume_fully_accessible = assume_fully_accessible
        self.access_vols = self.get_accessible_volumes(
            n_side=20, assume_fully_accessible=assume_fully_accessible
        )
        self.chi = chi
        self.vf_limit = vf_limit
        self.binder_dict = self.binders.to_dict(orient='records')
        self.fast_field = fast_field
        if fast_field == 1:
            self.init_fast_field(n_points=n_points)
        self.update_all_densities_for_all_polymers()
        self.affected_bins_last_move = np.zeros((self.n_bins,), dtype=int)
        self.dict_ = self.get_dict()

    def init_grid(self):
        """Initialize the discrete grid containing the field.

        Notes
        -----
        The field will be represented by a grid of rectangular prism (typically
        cubical) voxels, each of dimensions `dx` by `dy` by `dz`. To calculate
        interaction energies, density will be determined inside each voxel and
        a mean field approach will be applied.
        """
        self.dx = self.x_width / self.nx
        self.dy = self.y_width / self.ny
        self.dz = self.z_width / self.nz
        self.dxyz = np.array([self.dx, self.dy, self.dz])
        self.n_bins = self.nx * self.ny * self.nz
        self.vol_bin = self.x_width * self.y_width * self.z_width / self.n_bins
        self.bin_index = UniformDensityField._get_corner_bin_index(
            self.nx, self.ny, self.nz
        )
        self.width_xyz = np.array([self.x_width, self.y_width, self.z_width])
        self.half_width_xyz = np.array(
            [0.5 * self.x_width, 0.5 * self.y_width, 0.5 * self.z_width]
        )
        self.half_step_xyz = np.array(
            [0.5 * self.dx, 0.5 * self.dy, 0.5 * self.dz]
        )
        self.n_xyz_m1 = np.array(
            [self.nx - 1, self.ny - 1, self.nz - 1]
        )
        self.precompute_ind_xyz_to_super()
        self.nbr_inds = np.empty((8,), dtype=int)
        self.nbr_inds_with_trial = np.empty((2, 8), dtype=int)
        self.wt_vec = np.empty((8,), dtype=float)
        self.wt_vec_with_trial = np.empty((2, 8), dtype=float)
        self.xyz = np.empty((3,), dtype=float)
        self.xyz_with_trial = np.empty((2, 3), dtype=float)
        self.weight_xyz = np.empty((3,), dtype=float)
        self.weight_xyz_with_trial = np.empty((2, 3), dtype=float)
        self.index_xyz = np.empty((3,), dtype=int)
        self.index_xyz_with_trial = np.empty((2, 3), dtype=int)

    cdef void init_fast_field(self, long n_points):
        """Precompute grid of sub-bins for efficiently interpolating densities.
        
        Notes
        -----
        We will discretize the positions inside each voxel (forming "sub-bins") 
        and precompute the voxel weightings inside each sub-bin. We will map
        each sub-bin to a position in Cartesian space and to a pair of voxels
        splitting weight.
        
        With this discrete approach, we will no longer need to identify voxels
        containing beads and calculate associated weights from scratch during
        the inner loop of the MC algorithm. We will identify the x, y, z
        sub-bins containing a point and we will look up the pre-computed voxel
        weightings. This is expected to reduce the runtime of the field energy
        calculation, which accounts for the majority of the overall simulation
        runtime.
        
        This approach is only active when `fast_field` is set equal to `1` when
        initializing the uniform density field. It is recommended to use this
        ``fast field'' approach when simulations involve more than one
        epigenetic mark or full chromosomes.
        
        Parameters
        ----------
        n_points : long
            Number of sub-bins to precompute in each dimension when the ``fast
            field'' is active; should be an even number
        """
        cdef long i, j, k, ind
        cdef long half_n_points, lower_bin_ind, upper_bin_ind
        cdef double lower_weight, upper_weight

        n_points = n_points + (n_points % 2)
        self.n_points = n_points

        self.sub_bin_width_x = self.dx / n_points
        self.sub_bin_width_y = self.dy / n_points
        self.sub_bin_width_z = self.dz / n_points
        self.n_sub_bins_x = (n_points * self.nx)
        self.n_sub_bins_y = (n_points * self.ny)
        self.n_sub_bins_z = (n_points * self.nz)

        # Map sub-bins to bins; map sub_bins to weights
        self.sub_bins_to_bins_x = {}
        self.sub_bins_to_bins_y = {}
        self.sub_bins_to_bins_z = {}
        self.sub_bins_to_weights_x = {}
        self.sub_bins_to_weights_y = {}
        self.sub_bins_to_weights_z = {}

        for i in range(self.n_sub_bins_x):
            ind = \
                np.floor((i - n_points/2) * self.sub_bin_width_x / self.dxyz[0])
            if ind == -1:
                lower_bin_ind = self.nx - 1
            else:
                lower_bin_ind = ind
            upper_bin_ind = (ind + 1) % self.nx
            upper_weight = (
                ((i - n_points/2) * self.sub_bin_width_x) / self.dxyz[0] - ind
            )
            lower_weight = 1 - upper_weight
            self.sub_bins_to_bins_x[i] = [lower_bin_ind, upper_bin_ind]
            self.sub_bins_to_weights_x[i] = [lower_weight, upper_weight]

        for i in range(self.n_sub_bins_y):
            ind = \
                np.floor((i - n_points/2) * self.sub_bin_width_y / self.dxyz[1])
            if ind == -1:
                lower_bin_ind = self.ny - 1
            else:
                lower_bin_ind = ind
            upper_bin_ind = (ind + 1) % self.ny
            upper_weight = (
                ((i - n_points/2) * self.sub_bin_width_y) / self.dxyz[1] - ind
            )
            lower_weight = 1 - upper_weight
            self.sub_bins_to_bins_y[i] = [lower_bin_ind, upper_bin_ind]
            self.sub_bins_to_weights_y[i] = [lower_weight, upper_weight]

        for i in range(self.n_sub_bins_z):
            ind = \
                np.floor((i - n_points/2) * self.sub_bin_width_z / self.dxyz[2])
            if ind == -1:
                lower_bin_ind = self.nz - 1
            else:
                lower_bin_ind = ind
            upper_bin_ind = (ind + 1) % self.nz
            upper_weight = (
                ((i - n_points/2) * self.sub_bin_width_z) / self.dxyz[2] - ind
            )
            lower_weight = 1 - upper_weight
            self.sub_bins_to_bins_z[i] = [lower_bin_ind, upper_bin_ind]
            self.sub_bins_to_weights_z[i] = [lower_weight, upper_weight]

    cdef void precompute_ind_xyz_to_super(self):
        """Precompute how voxel (x, y, z) indices translate to superindices.
        """
        cdef long i, j, k
        self.inds_xyz_to_super = np.empty(
            (self.nx+1, self.ny+1, self.nz+1), dtype=int
        )
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                for k in range(self.nz + 1):
                    self.inds_xyz_to_super[i, j, k] = inds_to_super_ind(
                    i % self.nx, j % self.ny, k % self.nz, self.nx, self.ny
                )

    def init_field_energy_prefactors(self):
        """Initialize the field energy prefactor for each reader protein.
        """
        binder_names = []
        for i in range(self.num_binders):
            binder_name = self.binders.loc[i, "name"]
            binder_names.append(binder_name)
        for i in range(self.num_binders):
            self.binders.at[i, 'field_energy_prefactor'] = (
                0.5 * self.binders.iloc[i].interaction_energy
                * self.binders.iloc[i].interaction_volume
                * self.vol_bin
            )
            self.binders.at[i, 'interaction_energy_intranucleosome'] = (
                self.binders.iloc[i].interaction_energy
                * (1 - self.binders.iloc[i].interaction_volume / self.vol_bin)
            )
            for next_binder in binder_names:
                if next_binder in self.binders.iloc[i].cross_talk_interaction_energy.keys():
                    self.binders.at[i, 'cross_talk_field_energy_prefactor'][next_binder] = (
                        self.binders.iloc[i].cross_talk_interaction_energy[next_binder]
                        * self.binders.iloc[i].interaction_volume
                        * self.vol_bin
                    )
                else:
                    self.binders.at[i, 'cross_talk_field_energy_prefactor'][next_binder] = 0

    cpdef dict get_accessible_volumes(
            self, long n_side, bint assume_fully_accessible
    ):
        """Numerically find accessible volume of voxels at confinement edge.
        
        Notes
        -----
        Right now, accessible volume calculations are only relevant to spherical
        confinements.

        Parameters
        ----------
        n_side : long
            When a voxel is determined to be partially contained by the
            confinement, we will numerically determine the accessible volume in
            that voxel. To do this, we will place a cubic grid of points within
            the voxel, and we will count the number of grid points within the
            confinement. The parameter `n_side` gives the number of grid points
            to include on each side of the cubic grid. For example, if n_side =
            10, then we will place a 10x10x10 grid of points in every partially
            contained voxel to determine the accessible volume of that voxel.
        assume_fully_accessible : bint
            Flag indicating whether to assume all voxels are fully accessible.
            Assume voxels are fully accessible if the voxel volumes are far less
            than the confinement volume. Value of `1` indicates that all voxels
            are assumed to be fully accessible, bypassing the calculation of
            accessible volume.

        Returns
        -------
        access_vol : dict
            Mapping of bin super index to accessible volume inside the 
            confining boundary
        """
        cdef long i, voxel
        cdef long[:] split_voxels
        cdef long[:, ::1] xyz_inds
        cdef double[:, ::1] xyz_coords, dxyz_point
        cdef dict access_vols
        access_vols = {i : self.vol_bin for i in range(self.n_bins)}
        if assume_fully_accessible == 1:
            return access_vols
        if self.confine_type == "Spherical":
            xyz_inds = np.zeros((self.n_bins, 3), dtype=int)
            for i in range(self.n_bins):
                xyz_inds[i, :] = super_ind_to_inds(i, self.nx, self.ny)
            xyz_coords = self.get_voxel_coords(xyz_inds)
            buffer_dist = np.sqrt(2) / 4 * max(self.dx, self.dy, self.dz)
            split_voxels = self.get_split_voxels(xyz_coords, buffer_dist)
            dxyz_point = self.define_voxel_subgrid(n_side)
            for i in range(self.n_bins):
                if split_voxels[i] == 1:
                    access_vols[i] = (
                        self.vol_bin *
                        self.get_frac_accessible(xyz_coords[i], dxyz_point)
                    )
        return access_vols

    cdef double[:, ::1] get_voxel_coords(self, long[:, ::1] xyz_inds):
        """Get voxel coordinates from xyz bin indices.
        
        Notes
        -----
        Get the (x, y, z) coordinates at the center of each voxel. Start by
        converting voxel super-indices to x, y, and z indices. Then, multiply
        the x, y, and z indices by voxel widths dx, dy, and dz to get the
        coordinates at the center of each voxel.

        Parameters
        ----------
        xyz_inds : long[:, ::1]
            Indices in the x, y, and z directions for each bin, where rows
            represent individual bins and columns indicate the index position
            of the bin in the x, y, and z directions.

        Returns
        -------
        double[:, ::1]
            Coordinates in the x, y, and z directions for each bin, where rows
            represent individual bins and columns indicate coordinates in the
            x, y, and z directions.
        """
        cdef long i, j
        cdef long[:] nxyz
        cdef double[:, ::1] dxyz, centralized_inds, xyz_coords
        nxyz = np.array([self.nx, self.ny, self.nz])
        dxyz = np.array([
                [self.dx, 0, 0],
                [0, self.dy, 0],
                [0, 0, self.dz]
        ])
        centralized_inds = np.zeros((self.n_bins, 3))
        for i in range(self.n_bins):
            for j in range(3):
                centralized_inds[i, j] = xyz_inds[i, j] - (nxyz[j] - 1) / 2
        xyz_coords = np.matmul(centralized_inds, dxyz)
        return xyz_coords
    
    cdef long[:] get_split_voxels(
        self, double[:, ::1] xyz_coords, double buffer_dist
    ):
        """Identify voxels intersecting with the confinement boundary.
        
        Notes
        -----
        Flag each voxel as split to begin with. Then, for each voxel, calculate
        the distance from the origin to the center of the voxel. For voxels
        which fall within +/- one `buffer_dist` of the confining length, flag
        that voxel as split.
        
        Accessible volumes are only relevant for spherical confinements. In a
        cubical confinement, all voxels are entirely accessible.

        Parameters
        ----------
        xyz_coords : double[:, ::1]
            Cartesian coordinates identifying the center of each voxel, where
            rows represent individual voxels and columns indicate x, y, and z
            positions at the center of the voxel
        buffer_dist : double
            Distance from confining boundary at which a voxel is marked as 
            "split" and is assessed as potentially intersecting the boundary

        Returns
        -------
        long[:]
            Array of super-indices cooresponding to voxels that intersect with
            the confinement
        """
        cdef long[:] split_voxels = np.ones(self.n_bins, dtype=int)
        
        for i in range(self.n_bins):
            dist = sqrt(
                xyz_coords[i, 0] ** 2 +
                xyz_coords[i, 1] ** 2 +
                xyz_coords[i, 2] ** 2
            )
            if (dist < self.confine_length - buffer_dist or 
                dist > self.confine_length + buffer_dist):
                split_voxels[i] = 0
        return split_voxels
    
    cdef double[:, ::1] define_voxel_subgrid(self, long n_pt_side):
        """Define voxel subgrid for accessible volume calculation.
        
        Notes
        -----
        When a voxel is split by a confining boundary, we determine the
        accessible volume in that voxel using a numerical approach. We
        initialize an cubical array of points of dimensions `n_pt_side` by
        `n_pt_side` by `n_pt_side` inside the voxel. We then calculate the
        radial distance of each point in this array. We determine the fraction
        of points which fall inside the confinement, and we use this as an
        estimate of the fraction voxel volume that is accessible. We multiply
        this fraction by the full voxel volume to get the accessible volume in
        the voxel.   

        Parameters
        ----------
        n_pt_side : long
            When a voxel is determined to be partially contained by the
            confinement, we will numerically determine the accessible volume in
            that voxel. To do this, we will place a cubic grid of points within
            the voxel, and we will count the number of grid points within the
            confinement. The parameter `n_side` gives the number of grid points
            to include on each side of the cubic grid. For example, if n_side =
            10, then we will place a 10x10x10 grid of points in every partially
            contained voxel to determine the accessible volume of that voxel.

        Returns
        -------
        double[:, ::1]
            Array of subgrid points (relative to origin) for determination of
            voxel accessible volume. Rows indicate individual points, and 
            columns indicate x, y, z coordinates for each point.
        """
        cdef long total_points
        cdef double dx_point, dy_point, dz_point
        cdef double[:, ::1] dxyz_point

        total_points = n_pt_side ** 3
        dx_point = self.dx / n_pt_side
        dy_point = self.dy / n_pt_side
        dz_point = self.dz / n_pt_side
        dxyz_point = np.reshape(
            np.mgrid[
                0:n_pt_side,0:n_pt_side, 0:n_pt_side
            ].T, (total_points, 3)
        ).astype(float)
        for i in range(total_points):
            dxyz_point[i, 0] *= dx_point
            dxyz_point[i, 1] *= dy_point
            dxyz_point[i, 2] *= dz_point
        return dxyz_point

    cdef double get_frac_accessible(
        self, double[:] coords, double[:, ::1] dxyz_point
    ):
        """Get the fraction of a bin which is accessible inside a confinement.
        
        Notes
        -----
        See notes in `self.define_voxel_subgrid()` for implementation details.

        Parameters
        ----------
        coords : double[:]
            Cartesian coordinates in the form (x, y, z) identifying the center
            of the bin
        dxyz_point : double[:, ::1]
            Array of subgrid points (relative to origin) for determination of
            voxel accessible volume. Rows indicate individual points, and 
            columns indicate x, y, z coordinates for each point.

        Returns
        -------
        double
            Fraction of the bin which is accessible inside the confinement
        """
        cdef long i, n_points
        cdef double n_points_dbl, count_true, frac_accessible
        cdef double[:] dists
        cdef double[:, ::1] points

        n_points = dxyz_point.shape[0]
        n_points_dbl = float(n_points)
        points = dxyz_point.copy()
        for i in range(n_points):
            points[i, 0] += coords[0] - self.dx/2
            points[i, 1] += coords[1] - self.dy/2
            points[i, 2] += coords[2] - self.dz/2
        dists = np.linalg.norm(points, axis=1)
        count_true = 0.
        for i in range(n_points):
            if dists[i] < self.confine_length:
                count_true += 1
        frac_accessible = count_true / n_points_dbl
        return frac_accessible

    def to_file(self, path):
        """Save Field description + Polymer/ReaderProtein names to CSV.

        Parameters
        ----------
        path : str
            File path at which to save the CSV file representing the field

        Returns
        -------
        None or str
            Returns `None` if `path` is not `None`; otherwise returns the CSV
            representing the field as a string
        """
        rows = {name: self.dict_[name] for name in self._field_descriptors}
        for i, polymer in enumerate(self.polymers):
            rows[polymer.name] = 'polymer'
        for i, binder in self.binders.iterrows():
            # careful! binder.name is the Series.name attribute
            rows[binder['name']] = 'binder'
        # prints just key,value for each key in rows
        return pd.Series(rows).to_csv(path, header=False)

    @classmethod
    def from_file(cls, path, polymers, binders):
        """Recover field saved with `.to_file()`.

        Notes
        -----
        The zeroths column is an index column, and the first column stores the
        data specifying the field.

        Parameters
        ----------
        path : str
            Path to the CSV file representing the field
        polymers : List[PolymerBase]
            Polymers contained in the field
        binders : pd.DataFrame
            Table describing reader proteins affecting the field

        Returns
        -------
        UniformDensityField
            Discrete density field represented by the CSV file
        """
        field_series = pd.read_csv(path, header=None, index_col=0)[1]
        kwargs = field_series[_field_descriptors].to_dict()
        for key in kwargs.keys():
            if key in _int_field_descriptors:
                kwargs[key] = int(kwargs[key])
            elif key in _float_field_descriptors:
                kwargs[key] = float(kwargs[key])
            elif key in _str_field_descriptors and pd.isna(kwargs[key]):
                kwargs[key] = ""
            elif key in _bool_field_descriptors:
                kwargs[key] = (key == 'True')
        polymer_names = field_series[field_series == 'polymer'].index.values
        binder_names = field_series[field_series == 'binder'].index.values
        err_prefix = f"Tried to instantiate class:{cls} from file:{path} with "

        if len(polymers) != len(polymer_names):
            raise ValueError(
                err_prefix + f"{len(polymers)} polymers, but "
                f" there are {len(polymer_names)} listed."
            )
        for polymer in polymers:
            if polymer.name not in polymer_names:
                raise ValueError(
                    err_prefix + f"polymer:{polymer.name}, but "
                    " this polymer was not present in file."
                )
        if len(binders) != len(binder_names):
            raise ValueError(
                err_prefix + f"{len(binders)} binders, but "
                f" there are {len(binder_names)} listed."
            )
        for i, binder in binders.iterrows():
            if binder['name'] not in binder_names:
                raise ValueError(
                    err_prefix + f"binder:{binder}, but "
                    " this binder was not present in file."
                )
        return cls(polymers=polymers, binders=binders, **kwargs)

    def __eq__(self, other):
        """Check if two `UniformDensityField` objects are equivalent.

        Parameters
        ----------
        other : UniformDensityField
            Field being compared to the current field.

        Returns
        -------
        bool
            True if two fields are equivalent, else false.
        """
        return np.all(
            [
                self.polymers == other.polymers,
                self.binders.equals(other.binders),
                self.nx == other.nx,
                self.x_width == other.x_width,
                self.ny == other.ny,
                self.y_width == other.y_width,
                self.nz == other.nz,
                self.z_width == other.z_width
            ]
        )

    @staticmethod
    def _get_corner_bin_index(nx, ny, nz) -> np.ndarray:
        """Set up the index array for density bins corners.

        Notes
        -----
        TODO: this should just be a higher-dimensional array to avoid having to
        do any of this math.

        Parameters
        ----------
        nx, ny, nz :  long
            Number of voxels in the x, y, and z direction of the field

        Returns
        -------
        array_like (nx*ny*nz, 8) of long
            For each of the `nx*ny*nz` voxels in the field, stores indices
            identifying eight vertices at the corners of the voxel
        """
        bin_index = np.zeros((nx * ny * nz, 8), dtype=int)
        count = 0
        for index_z in range(nz):
            if index_z == nz - 1:
                index_zp1 = 0
            else:
                index_zp1 = index_z + 1
            for index_y in range(ny):
                if index_y == ny - 1:
                    index_yp1 = 0
                else:
                    index_yp1 = index_y + 1
                for index_x in range(nx):
                    if index_x == nx - 1:
                        index_xp1 = 0
                    else:
                        index_xp1 = index_x + 1
                    # Populate the bin_index array with the 8 corner bins
                    bin_index[count] = [
                        index_x + nx * index_y + nx * ny * index_z,
                        index_xp1 + nx * index_y + nx * ny * index_z,
                        index_x + nx * index_yp1 + nx * ny * index_z,
                        index_xp1 + nx * index_yp1 + nx * ny * index_z,
                        index_x + nx * index_y + nx * ny * index_zp1,
                        index_xp1 + nx * index_y + nx * ny * index_zp1,
                        index_x + nx * index_yp1 + nx * ny * index_zp1,
                        index_xp1 + nx * index_yp1 + nx * ny * index_zp1
                    ]
                    count += 1
        return bin_index

    def __str__(self):
        """Print summary of the UniformDensityField.
        """
        n_poly = len(self.polymers)
        return f"UniformDensityField<nx={self.nx},ny={self.ny},nz={self.nz}," \
               f"npoly={n_poly}>"

    def get_dict(self):
        """Dictionary representation of the `UniformDensityField` object.

        Returns
        -------
        dict
            Dictionary of key attributes representing the field
        """
        return {
            "x_width" : self.x_width,
            "y_width" : self.y_width,
            "z_width" : self.z_width,
            "nx" : self.nx,
            "ny" : self.ny,
            "nz" : self.nz,
            "num_binders" : self.num_binders,
            "n_bins" : self.n_bins,
            "density" : self.density,
            "confine_type" : self.confine_type,
            "confine_length" : self.confine_length,
            "chi" : self.chi,
            "assume_fully_accessible": self.assume_fully_accessible,
            "vf_limit": self.vf_limit,
            "fast_field": self.fast_field,
            "n_points": self.n_points
        }

    cdef double compute_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long packet_size, bint state_change
    ):
        """Compute field contribution of energy change for proposed MC move.

        Notes
        -----
        For polymer affected by a move, compare the field energy of the new 
        and old configurations.

        First verify all points lie inside the specified confining boundary.
        If a point lies outside the boundary, return a high energy to reject
        the move and stop field energy calculation. If all beads affected by
        the move lie inside the confinement, do the following:

        Load the current position and states for the polymer. Using this data,
        identify the density of the polymer and binders in each bin in space.
        Repeat this density calculation for the trial polymer and binders
        configuration. Determine the self-interaction energy based on the
        interpolated densities between each bin using the approach described
        by MacPherson et al [1]. For each binder on the chain, add its
        contribution to the field energy based on interactions with like
        binders.

        References
        ----------
        [1] MacPherson, Q.; Beltran, B.; Spakowitz, A. J. Bottom–up Modeling of
            Chromatin Segregation Due to Epigenetic Modifications. PNAS 2018,
            115 (50), 12739–12744. https://doi.org/10.1073/pnas.1812268115.

        Parameters
        ----------
        poly : `chromo.PolymerBase`
            The polymer which has been moved.
        inds : array_like (N,)
            Indices of beads being moved
        n_inds : long
            Number of beads affected by the move
        packet_size : long
            Number of points to average together when calculating the field
            energy change of a move; done to reduce the computational expense
            of the field energy calculation (at the expense of precision).
            Currently this does nothing. Packet size is a consideration for
            later on if the simulation is too slow.
        state_change : bint
            Indicator for whether the MC move involved a change in binding state
            (1) or not (0)

        Returns
        -------
        dE : double
            Change in field energy assocaited with the change in polymer 
            configuration
        """
        cdef long i, ind, bin_ind
        cdef long[:] delta_ind_xyz, bin_inds
        cdef double dE = 0

        # Verify that move does not violate the hard confinement
        if state_change == 0:
            dE += self.get_confinement_dE(poly, inds, n_inds, trial=1)
            dE -= self.get_confinement_dE(poly, inds, n_inds, trial=0)

        # Find changes in polymer density in affected voxels
        if self.fast_field == 1:
            bin_inds = self.get_change_in_density_quickly(
                poly, inds, n_inds, state_change
            )
        else:
            bin_inds = self.get_change_in_density(
                poly, inds, n_inds, state_change
            )

        for i in range(self.n_bins):
            self.affected_bins_last_move[i] = 0
        for bin_ind in bin_inds:
            self.affected_bins_last_move[bin_ind] = 1

        # Get change in energy based on differences in bead and binder densities
        dE += self.get_dE_binders_and_beads(
            poly, inds, n_inds, bin_inds, state_change
        )

        return dE

    cdef long[:] get_change_in_density_quickly(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        bint state_change
    ):
        """Calculate the density in bins affected by the MC move.
        
        Notes
        -----
        See notes for `self.get_change_in_density()`. Uses pre-computed sub-bin
        weightings. 
        """
        cdef long i, j, k, l, m, sub_bin_x, sub_bin_y, sub_bin_z,
        cdef long sub_bin_x_trial, sub_bin_y_trial, sub_bin_z_trial
        cdef double prefactor
        cdef double[:, :, ::1] densities
        cdef set bins_found
        bins_found = set()

        for i in range(n_inds):

            # Get sub-bin indices
            sub_bin_x = np.floor(
                (poly.r[inds[i], 0] + self.half_width_xyz[0]) /
                self.sub_bin_width_x
            ) % self.n_sub_bins_x
            sub_bin_y = np.floor(
                (poly.r[inds[i], 1] + self.half_width_xyz[1]) /
                self.sub_bin_width_y
            ) % self.n_sub_bins_y
            sub_bin_z = np.floor(
                (poly.r[inds[i], 2] + self.half_width_xyz[2]) /
                self.sub_bin_width_z
            ) % self.n_sub_bins_z

            # Identify sub-bin weights in x, y, z directions
            self.weight_xyz_with_trial[0, 0] = \
                self.sub_bins_to_weights_x[sub_bin_x][0]
            self.weight_xyz_with_trial[0, 1] = \
                self.sub_bins_to_weights_y[sub_bin_y][0]
            self.weight_xyz_with_trial[0, 2] = \
                self.sub_bins_to_weights_z[sub_bin_z][0]

            if state_change == 0:

                sub_bin_x_trial = np.floor(
                    (poly.r_trial[inds[i], 0] + self.half_width_xyz[0]) /
                    self.sub_bin_width_x
                ) % self.n_sub_bins_x
                sub_bin_y_trial = np.floor(
                    (poly.r_trial[inds[i], 1] + self.half_width_xyz[1]) /
                    self.sub_bin_width_y
                ) % self.n_sub_bins_y
                sub_bin_z_trial = np.floor(
                    (poly.r_trial[inds[i], 2] + self.half_width_xyz[2]) /
                    self.sub_bin_width_z
                ) % self.n_sub_bins_z

            else:
                sub_bin_x_trial = sub_bin_x
                sub_bin_y_trial = sub_bin_y
                sub_bin_z_trial = sub_bin_z
                for j in range(3):
                    self.xyz_with_trial[1, j] = self.xyz_with_trial[0, j]

            self.weight_xyz_with_trial[1, 0] = \
                self.sub_bins_to_weights_x[sub_bin_x_trial][0]
            self.weight_xyz_with_trial[1, 1] = \
                self.sub_bins_to_weights_y[sub_bin_y_trial][0]
            self.weight_xyz_with_trial[1, 2] = \
                self.sub_bins_to_weights_z[sub_bin_z_trial][0]

            self.index_xyz_with_trial[0, 0] = \
                self.sub_bins_to_bins_x[sub_bin_x][0]
            self.index_xyz_with_trial[0, 1] = \
                self.sub_bins_to_bins_y[sub_bin_y][0]
            self.index_xyz_with_trial[0, 2] = \
                self.sub_bins_to_bins_z[sub_bin_z][0]
            self.index_xyz_with_trial[1, 0] = \
                self.sub_bins_to_bins_x[sub_bin_x_trial][0]
            self.index_xyz_with_trial[1, 1] = \
                self.sub_bins_to_bins_y[sub_bin_y_trial][0]
            self.index_xyz_with_trial[1, 2] = \
                self.sub_bins_to_bins_z[sub_bin_z_trial][0]

            # Get weights and super-indices of eight bins containing bead
            self._generate_weight_vector_with_trial()
            self._generate_index_vector_with_trial()

            # Distribute weights into bins
            # k indicates current (0) or trial (1) configuration
            # l indicates for which of eight bins density is being calculated
            # m indicates polymer (0) or which protein (1 to n_binders)
            for k in range(2):
                for l in range(8):
                    poly.densities_temp[k, 0, l] = (
                            self.wt_vec_with_trial[k, l] /
                            self.access_vols[self.nbr_inds_with_trial[k, l]]
                    )
                    for m in range(1, poly.n_binders_p1):
                        # Current Configuration
                        if k == 0 or state_change == 0:
                            poly.densities_temp[k, m, l] = \
                                poly.densities_temp[k, 0, l] * \
                                float(poly.states[inds[i], m-1])
                        # Trial Configuration of State Change
                        else:
                            poly.densities_temp[k, m, l] = \
                                poly.densities_temp[k, 0, l] * \
                                float(poly.states_trial[inds[i], m-1])

            # Combine repeating elements, stored in poly.density_trial
            for k in range(2):
                # Subtract if current configuration
                if k == 0:
                    prefactor = -1.
                else:
                    prefactor = 1.

                # Accumulate densities
                for l in range(8):
                    if self.nbr_inds_with_trial[k, l] in bins_found:
                        for m in range(poly.n_binders_p1):
                            temp = prefactor * poly.densities_temp[k, m, l]
                            self.density_trial[
                                self.nbr_inds_with_trial[k, l], m
                            ] += temp
                    else:
                        bins_found.add(self.nbr_inds_with_trial[k, l])
                        for m in range(poly.n_binders_p1):
                            temp = prefactor * poly.densities_temp[k, m, l]
                            self.density_trial[
                                self.nbr_inds_with_trial[k, l], m
                            ] = temp
        return np.array(list(bins_found))

    cdef long[:] get_change_in_density(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        bint state_change
    ):
        """Calculate the density in bins affected by the MC move.

        Notes
        -----
        This method takes in a polymer configuration (either current or trial,
        as dectated by the `trial` argument) and updates a field density map.
        The method operates by the following steps:

        (1) Begin by generating weights and indices affected by the MC move.

            - Start by identifying the (0, 0, 0) bins for the affected beads
            and the associated weights.

            - Then generate a weight array for all bins containing affected
            beads.

            - Finally, generate an index vector identifying the affected bins.
        
        (2) Then initialize a density vector and loop through binders and
        corner bin indices to fill in the density vector.

            - Values in arrays are determined element-by-element to improve
            speed once compiled.

        (3) Combine repeated entries in the density array, and return the
        affected densities and bin indices.

        TODO: Replace dict with set (once I figure out how to get that working)

        NOTE: Replacing k-loops with direct statements improves speed by approx
        0.7% -- we opt for clean code instead.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N, ) of long
            Ordered indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move
        state_change : bint
            Indicator for whether the MC move involved a change in binding state
            (1) or not (0)

        Returns
        -------
        long[:]
            Array of unique bin indices affected by the proposed MC move
        """
        cdef long i, j, k, l, m, ind
        cdef double prefactor
        cdef double[:, :, ::1] densities

        cdef set bins_found
        bins_found = set()

        for i in range(n_inds):
            # Shift positions so all positive, apply periodic boundaries
            # Get the lower neighboring bin index

            for j in range(3):
                # Load current and trial configuration of the polymer (current
                # in row 0; trial in row 1)
                self.xyz_with_trial[0, j] = (
                    (poly.r[inds[i], j] + self.half_width_xyz[j]) %
                    self.width_xyz[j]
                ) - self.half_step_xyz[j]
                if state_change == 0:
                    self.xyz_with_trial[1, j] = (
                        (poly.r_trial[inds[i], j] + self.half_width_xyz[j]) %
                        self.width_xyz[j]
                    ) - self.half_step_xyz[j]
                else:
                    self.xyz_with_trial[1, j] = self.xyz_with_trial[0, j]

                # Get the lower neighboring bin index
                for k in range(2):
                    ind = <long>floor(
                        (self.xyz_with_trial[k, j]) / self.dxyz[j]
                    )
                    if ind == -1:
                        self.index_xyz_with_trial[k, j] = self.n_xyz_m1[j]
                    else:
                        self.index_xyz_with_trial[k, j] = ind
                
                    # Get weight in the lower bin index
                    self.weight_xyz_with_trial[k, j] = (
                        1 - (self.xyz_with_trial[k, j] / self.dxyz[j] - ind)
                    )

            # Get weights and superindices of eight bins containing bead
            self._generate_weight_vector_with_trial()
            self._generate_index_vector_with_trial()

            # Distribute weights into bins
            # k indicates current (0) or trial (1) configuration
            # l indicates for which of eight bins density is being calculated
            # m indicates polymer (0) or which protein (1 to n_binders)
            for k in range(2):
                for l in range(8):
                    poly.densities_temp[k, 0, l] = (
                        self.wt_vec_with_trial[k, l] /
                        self.access_vols[self.nbr_inds_with_trial[k, l]]
                    )
                    for m in range(1, poly.n_binders_p1):
                        # Current Configuration
                        if k == 0 or state_change == 0:
                            poly.densities_temp[k, m, l] =\
                                poly.densities_temp[k, 0, l] *\
                                float(poly.states[inds[i], m-1])
                        # Trial Configuration of State Change
                        else:
                            poly.densities_temp[k, m, l] =\
                                poly.densities_temp[k, 0, l] *\
                                float(poly.states_trial[inds[i], m-1])

            # Combine repeating elements, stored in poly.density_trial
            for k in range(2):
                # Subtract if current configuration
                if k == 0:
                    prefactor = -1.
                else:
                    prefactor = 1.

                # Accumulate densities
                for l in range(8):
                    if self.nbr_inds_with_trial[k, l] in bins_found:
                        for m in range(poly.n_binders_p1):
                            temp = prefactor * poly.densities_temp[k, m, l]
                            # Fix rounding error
                            if np.abs(temp) > 1E-18:
                                self.density_trial[
                                    self.nbr_inds_with_trial[k, l], m
                                ] += temp
                    else:
                        bins_found.add(self.nbr_inds_with_trial[k, l])
                        for m in range(poly.n_binders_p1):
                            temp = prefactor * poly.densities_temp[k, m, l]
                            # Fix rounding error
                            if np.abs(temp) > 1E-18:
                                self.density_trial[
                                    self.nbr_inds_with_trial[k, l], m
                                ] = temp
                            else:
                                self.density_trial[
                                    self.nbr_inds_with_trial[k, l], m
                                ] = 0

        return np.array(list(bins_found))

    cdef void _generate_weight_vector_with_trial(self):
        """Generate weight array for eight bins containing the bead.
        
        Notes
        -----
        The weight vector contains the fraction of a point's weight distributed
        to each of the eight voxels containing the point. To determine weights,
        we calculating distances between a point and the centers of the two
        nearest voxels containing it in each the x, y, and z directions. The
        weight is determined from these distances using the lever rule.
        
        The values stored in `self.weight_xyz_with_trial` indicate the weights
        in the *lower* voxels contianing the point, in each the x, y, and z
        directions. The first dimension of `self.weight_xyz_with_trial`
        indicates whether we are working with a current configuration (row 0)
        or trial configuration (row 1).
         
        """
        cdef int i
        for i in range(2):

            # lower_x, lower_y, lower_z
            self.wt_vec_with_trial[i, 0] = (
                self.weight_xyz_with_trial[i, 0] *
                self.weight_xyz_with_trial[i, 1] *
                self.weight_xyz_with_trial[i, 2]
            )

            # upper_x, lower_y, lower_z
            self.wt_vec_with_trial[i, 1] = (
                (1-self.weight_xyz_with_trial[i, 0]) *
                self.weight_xyz_with_trial[i, 1] *
                self.weight_xyz_with_trial[i, 2]
            )

            # lower_x, upper_y, lower_z
            self.wt_vec_with_trial[i, 2] = (
                self.weight_xyz_with_trial[i, 0] *
                (1-self.weight_xyz_with_trial[i, 1]) *
                self.weight_xyz_with_trial[i, 2]
            )

            # upper_x, upper_y, lower_z
            self.wt_vec_with_trial[i, 3] = (
                (1-self.weight_xyz_with_trial[i, 0]) *
                (1-self.weight_xyz_with_trial[i, 1]) *
                self.weight_xyz_with_trial[i, 2]
            )

            # lower_x, lower_y, upper_z
            self.wt_vec_with_trial[i, 4] = (
                self.weight_xyz_with_trial[i, 0] *
                self.weight_xyz_with_trial[i, 1] *
                (1-self.weight_xyz_with_trial[i, 2])
            )

            # upper_x, lower_y, upper_z
            self.wt_vec_with_trial[i, 5] = (
                (1-self.weight_xyz_with_trial[i, 0]) *
                self.weight_xyz_with_trial[i, 1] *
                (1-self.weight_xyz_with_trial[i, 2])
            )

            # lower_x, upper_y, upper_z
            self.wt_vec_with_trial[i, 6] = (
                self.weight_xyz_with_trial[i, 0] *
                (1-self.weight_xyz_with_trial[i, 1]) *
                (1-self.weight_xyz_with_trial[i, 2])
            )
            # upper_x, upper_y, upper_z
            self.wt_vec_with_trial[i, 7] = (
                (1-self.weight_xyz_with_trial[i, 0]) *
                (1-self.weight_xyz_with_trial[i, 1]) *
                (1-self.weight_xyz_with_trial[i, 2])
            )

    cdef void _generate_index_vector_with_trial(self):
        """Generate vector of eight superindices containing a bead.

        Notes
        -----
        This is a cleaner form of the method, but is ~70% slower:

        for i in range(2):
            for j in range(8):
                ind_x = ((j % 2) + index_xyz[i, 0]) % self.nx
                ind_y = (((int(floor(j/2))) % 2) + index_xyz[i, 1]) % self.ny
                ind_z = (((int(floor(j/4))) % 2) + index_xyz[i, 2]) % self.nz
                self.nbr_inds_with_trial[i, j] = inds_to_super_ind(
                    ind_x, ind_y, ind_z, self.nx, self.ny
                )
        """
        cdef int i
        for i in range(2):

            # lower_x, lower_y, lower_z
            self.nbr_inds_with_trial[i, 0] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]

            # upper_x, lower_y, lower_z
            self.nbr_inds_with_trial[i, 1] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]

            # lower_x, upper_y, lower_z
            self.nbr_inds_with_trial[i, 2] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]

            # upper_x, upper_y, lower_z
            self.nbr_inds_with_trial[i, 3] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]

            # lower_x, lower_y, upper_z
            self.nbr_inds_with_trial[i, 4] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]

            # upper_x, lower_y, upper_z
            self.nbr_inds_with_trial[i, 5] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]

            # lower_x, upper_y, upper_z
            self.nbr_inds_with_trial[i, 6] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]

            # upper_x, upper_y, upper_z
            self.nbr_inds_with_trial[i, 7] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]

    cdef double get_dE_binders_and_beads(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long[:] bin_inds, bint state_change
    ):
        """Get the change in energy associated with reconfiguration of binders.

        Notes
        -----
        The scale of the oligomerization energy is constant for each reader
        protein. While it does not affect runtime too much, this method can be
        optimized by precomputing the scale.

        TODO: Remove enumerate call -- that carries ogverhead

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N, ) of long
            Indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move
        bin_inds : long[:]
            Bin indices affected by the MC move either through loss or addition
            of density
        state_change : bint
            Indicator for whether the MC move involved a change in binding state
            (1) or not (0)

        Returns
        -------
        double
            Change in energy associated with reconfiguration of binders based
            on binder interaction energies, as well as nonspecific bead
            interactions
        """
        cdef long n_bins, i, j, kn_double_bound
        cdef double tot_density_change, dE_binders_beads
        cdef double[:, ::1] delta_rho_squared
        cdef double[:, :, ::1] delta_rho_interact_squared
        cdef dict binder_info

        # Change in squared density for each binder
        n_bins = len(bin_inds)
        delta_rho_squared = np.zeros((n_bins, poly.num_binders), dtype=float)
        delta_rho_interact_squared = np.zeros(
            (n_bins, poly.num_binders, poly.num_binders), dtype=float
        )
        for i in range(n_bins):
            for j in range(poly.num_binders):

                delta_rho_squared[i, j] = (
                    self.density[bin_inds[i], j+1] +
                    self.density_trial[bin_inds[i], j+1]
                ) ** 2 - self.density[bin_inds[i], j+1] ** 2

                for k in range(poly.num_binders):
                    delta_rho_interact_squared[i, j, k] = ((
                        self.density[bin_inds[i], j+1] +
                        self.density_trial[bin_inds[i], j+1]
                    ) * (
                        self.density[bin_inds[i], k+1] +
                        self.density_trial[bin_inds[i], k+1]
                    )) - (
                        self.density[bin_inds[i], j+1] *
                        self.density[bin_inds[i], k+1]
                    )

        # Fix rounding error
        for i in range(n_bins):
            for j in range(poly.num_binders):
                if np.abs(delta_rho_squared[i, j]) < 1E-18:
                    delta_rho_squared[i, j] = 0
                for k in range(poly.num_binders):
                    if np.abs(delta_rho_interact_squared[i, j, k]) < 1E-18:
                        delta_rho_interact_squared[i, j, k] = 0

        # Count num. nucleosomes bound by reader proteins at both histone tails
        self.count_doubly_bound(
            poly, inds, n_inds, trial=1, state_change=state_change
        )
        self.count_doubly_bound(
            poly, inds, n_inds, trial=0, state_change=state_change
        )

        dE_binders_beads = 0.0
        for i, binder_info in enumerate(self.binder_dict):
            # Calculate total density change
            tot_density_change = 0.0
            for j in range(n_bins):
                tot_density_change += delta_rho_squared[j, i]

            # Oligomerization
            dE_binders_beads +=\
                binder_info['field_energy_prefactor'] * tot_density_change

            # Intranucleosome interaction
            n_double_bound = self.doubly_bound_trial[i] - self.doubly_bound[i]
            dE_binders_beads +=\
                binder_info['interaction_energy_intranucleosome'] * n_double_bound

        # Cross-talk Interaction
        for i, binder_info in enumerate(self.binder_dict):
            for j, next_binder_info in enumerate(self.binder_dict):
                tot_density_change_interact = 0
                for k in range(n_bins):
                    tot_density_change_interact += delta_rho_interact_squared[k, i, j]

                dE_binders_beads +=\
                    binder_info["cross_talk_field_energy_prefactor"][next_binder_info["name"]] *\
                    tot_density_change_interact

        # Nonspecific bead interaction energy
        dE_binders_beads += self.nonspecific_interact_dE(poly, bin_inds, n_bins)

        return dE_binders_beads

    cdef double nonspecific_interact_dE(
        self, poly.PolymerBase poly, long[:] bin_inds, long n_bins
    ):
        """Get nonspecific interaction energy for an affected polymer segment.

        Notes
        -----
        This method assumes that all beads are of the same volume.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer affected by the current MC move
        bin_inds : long[:]
            Bin indices affected by the MC move either through loss or addition
            of density
        n_bins : long
            Number of bins affected by an MC move

        Returns
        -------
        double
            Change in energy associated with the nonspecific interactions
            between segments of the polymer chain
        """
        cdef double bead_V, nonspecific_dE, access_vol
        cdef double[:, ::1] vol_fracs

        bead_V = poly.beads[0].vol
        vol_fracs = self.get_volume_fractions_with_trial(
            bead_V, bin_inds, n_bins
        )
        nonspecific_dE = 0
        for i in range(n_bins):
            access_vol = self.access_vols[bin_inds[i]]
            
            # Trial Volume Fractions
            if vol_fracs[1, i] > self.vf_limit:
                nonspecific_dE += E_HUGE * vol_fracs[1, i]
            else:
                nonspecific_dE += self.chi * (access_vol / bead_V) *\
                                  vol_fracs[1, i] ** 2

            # Current volume fractions
            if vol_fracs[0, i] > self.vf_limit:
                nonspecific_dE -= E_HUGE * vol_fracs[0, i]
            else:
                nonspecific_dE -= self.chi * (access_vol / bead_V) *\
                                  vol_fracs[0, i] ** 2

        return nonspecific_dE

    cdef double[:, ::1] get_volume_fractions_with_trial(
        self, double bead_V, long[:] bin_inds, long n_bins
    ):
        """Calculate the volume frac. of beads in each bin of discretized space.

        Parameters
        ----------
        bead_V : double
            Volume of an individual bead
        bin_inds : long[:]
            Bin indices affected by the MC move either through loss or addition
            of density
        n_bins : long
            Number of bins affected by an MC move

        Returns
        -------
        double[:, ::1]
            Array of bead volume fractions for each bin; first row corresponds
            to current configuration of the polymer, and the second row
            corresponds to the trial configuration of the polymer
        """
        cdef long i, j
        cdef double density, change_in_density
        cdef double[:, ::1] vol_fracs = np.empty((2, n_bins), dtype=float)

        for j in range(n_bins):
            density = self.density[bin_inds[j], 0]
            change_in_density = self.density_trial[bin_inds[j], 0]
            vol_fracs[0, j] = density * bead_V
            vol_fracs[1, j] = vol_fracs[0, j] + (change_in_density * bead_V)
        return vol_fracs

    cdef void count_doubly_bound(
        self, poly.PolymerBase poly, long[:] inds, long n_inds, bint trial,
        bint state_change
    ):
        """For each reader protein, count the number of doubly bound beads.

        Notes
        -----
        We refer to nucleosomes with two histone tails bound by a reader
        protein as being doubly bound.

        When both histone tails of a nucleosome are bound by the same
        reader protein, there is an additional intranucleosome interaction
        energy contributing to the binding energy.

        This function counts the number of doubly bounded nucleosomes for each
        reader protein.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N,) of long
            Indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move
        trial : bint
            Indicator for whether to count doubly bound beads for the current
            configuration (0) or trial configuration (0)
        state_change : bint
            Indicator for whether the MC move involved a change in binding state
            (1) or not (0)
        """
        cdef long i, j

        # Count doubly-bound beads for current configuration
        if trial == 0:
            for j in range(poly.num_binders):
                self.doubly_bound[j] = 0
                for i in range(n_inds):
                    if poly.states[inds[i], j] == 2:
                        self.doubly_bound[j] += 1

        # Count doubly-bound beads for trial configuration
        elif trial == 1:
            if state_change == 1:
                for j in range(poly.num_binders):
                    self.doubly_bound_trial[j] = 0
                    for i in range(n_inds):
                        if poly.states_trial[inds[i], j] == 2:
                            self.doubly_bound_trial[j] += 1
            else:
                for j in range(poly.num_binders):
                    self.doubly_bound_trial[j] = 0
                    for i in range(n_inds):
                        if poly.states[inds[i], j] == 2:
                            self.doubly_bound_trial[j] += 1

        # Raise error if invalid `trial` flag is passed
        else:
            raise ValueError("Invalid current/trial state indicator.")

    cpdef double compute_E(self, poly.PolymerBase poly):
        """Compute total field energy for the current polymer configuration.
        
        Notes
        -----
        Load the positions and states of the entire polymer. Then determine the
        number density of beads in each bin of the field. Use these densities
        and binder states on each bead to determine an overall field energy.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field

        Returns
        -------
        double
            Total energy associated with the field
        """
        cdef long n_inds
        cdef double E
        cdef long[:] inds

        n_inds = poly.num_beads
        inds = np.arange(0, n_inds, 1)
        self.update_all_densities(poly, inds, n_inds)
        E = self.get_E_binders_and_beads(poly, inds, n_inds)
        return E

    cdef void update_affected_densities(self):
        """Update densities in affected bins when a move is accepted.
        """
        for i in range(self.n_bins):
            if self.affected_bins_last_move[i] == 1:
                for j in range(self.num_binders+1):
                    self.density[i, j] += self.density_trial[i, j]
                    self.density_trial[i, j] = 0

    cpdef void update_all_densities(
        self, poly.PolymerBase poly, long[:]& inds, long n_inds
    ):
        """Update the density of the field for a single polymer.
        
        Notes
        -----
        Updates the voxel densities stored in the field object. See notes for
        `self.get_change_in_density()` for details on implementation.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer for which densities are calculated
        inds : long[:] by reference
            Array of bead indices for the entire polymer (to avoid re-computing
            this)
        n_inds : long
            Number of beads in the polymer
        """
        cdef double density
        cdef long i, j, l, m, ind
        cdef long[:] superindices

        # Re-initialize all densities
        for i in range(self.n_bins):
            for j in range(poly.num_binders+1):
                self.density[i, j] = 0
                self.density_trial[i, j] = 0
        
        # Iterate through beads and add their densities to corresponding bins
        for i in range(n_inds):
            for j in range(3):
                
                # Load current configuration of the polymer
                self.xyz[j] = (
                    (poly.r[inds[i], j] + self.half_width_xyz[j]) %
                    self.width_xyz[j]
                ) - self.half_step_xyz[j]
                
                # Get the lower neighboring bin index
                ind = <long>floor((self.xyz[j]) / self.dxyz[j])
                if ind == -1:
                    self.index_xyz[j] = self.n_xyz_m1[j]
                else:
                    self.index_xyz[j] = ind

                # Get weight in the lower bin index
                self.weight_xyz[j] = (1 - (self.xyz[j] / self.dxyz[j] - ind))
            
            # Get weights and superindices of eight bins containing beads
            self._generate_weight_vector()
            self._generate_index_vector()

            # Distribute weights into bins
            # l indicates for which of eight bins density is being calculated
            # m indicates polymer (0) or which protein (1 to n_binders)
            for l in range(8):
                density = self.wt_vec[l] / self.access_vols[self.nbr_inds[l]]
                self.density[self.nbr_inds[l], 0] += density
                for m in range(1, poly.n_binders_p1):
                    self.density[self.nbr_inds[l], m] += density *\
                        float(poly.states[inds[i], m-1])

    cpdef void update_all_densities_for_all_polymers(self):
        """Update the density map for every polymer in the field.

        Notes
        -----
        Updates the voxel densities stored in the field object. See notes for
        `self.get_change_in_density()` for details on implementation.
        
        Requires that binders are listed in the same order on each polymer, as
        listed in `self.binders`.
        """
        cdef double density
        cdef long h, i, j, l, m, ind, n_binders_p1, n_inds, max_binder_count
        cdef long[:] superindices, inds, binder_counts
        cdef poly.PolymerBase poly

        # Re-initialize all densities and trial densities
        for i in range(self.n_bins):
            for j in range(self.num_binders):
                self.density[i, j] = 0
                self.density_trial[i, j] = 0

        for h in range(self.n_polymers):
            poly = self.polymers[h]
            inds = np.arange(poly.num_beads)
            n_inds = poly.num_beads
            n_binders_p1 = poly.n_binders_p1
            
            # Iterate through beads and add densities to corresponding bins
            for i in range(n_inds):
                for j in range(3):
                    
                    # Load current configuration of the polymer
                    self.xyz[j] = (
                        (poly.r[inds[i], j] + self.half_width_xyz[j]) %
                        self.width_xyz[j]
                    ) - self.half_step_xyz[j]
                    
                    # Get the lower neighboring bin index
                    ind = <long>floor((self.xyz[j]) / self.dxyz[j])
                    if ind == -1:
                        self.index_xyz[j] = self.n_xyz_m1[j]
                    else:
                        self.index_xyz[j] = ind

                    # Get weight in the lower bin index
                    self.weight_xyz[j] = (1 - (self.xyz[j] / self.dxyz[j] - ind))
                # Get weights and superindices of eight bins containing beads
                self._generate_weight_vector()
                self._generate_index_vector()

                # Distribute weights into bins
                # l indicates for which of 8 bins density is being calculated
                # m indicates polymer (0) or which protein (1 to n_binders)
                for l in range(8):
                    density = self.wt_vec[l] / self.access_vols[self.nbr_inds[l]]
                    self.density[self.nbr_inds[l], 0] += density
                    for m in range(1, n_binders_p1):
                        self.density[self.nbr_inds[l], m] += density *\
                            float(poly.states[inds[i], m-1])

        # Fix rounding error
        for i in range(self.n_bins):
            for j in range(self.num_binders + 1):
                if np.abs(self.density[i, j]) < 1E-18:
                    self.density[i, j] = 0
        
    cdef void _generate_weight_vector(self):
        """Generate weight array for eight bins containing the bead.
        
        Notes
        -----
        See notes in `self.generate_weight_vector_with_trial` for details on
        implementation.
        """
        self.wt_vec = np.array([

            # lower_x, lower_y, lower_z
            self.weight_xyz[0] * self.weight_xyz[1] * self.weight_xyz[2],

            # upper_x, lower_y, lower_z
            (1-self.weight_xyz[0]) * self.weight_xyz[1] *
            self.weight_xyz[2],

            # lower_x, upper_y, lower_z
            self.weight_xyz[0] * (1-self.weight_xyz[1]) *
            self.weight_xyz[2],

            # upper_x, upper_y, lower_z
            (1-self.weight_xyz[0]) * (1-self.weight_xyz[1]) *
            self.weight_xyz[2],

            # lower_x, lower_y, upper_z
            self.weight_xyz[0] * self.weight_xyz[1] *
            (1-self.weight_xyz[2]),

            # upper_x, lower_y, upper_z
            (1-self.weight_xyz[0]) * self.weight_xyz[1] *
            (1-self.weight_xyz[2]),

            # lower_x, upper_y, upper_z
            self.weight_xyz[0] * (1-self.weight_xyz[1]) *
            (1-self.weight_xyz[2]),

            # upper_x, upper_y, upper_z
            (1-self.weight_xyz[0]) * (1-self.weight_xyz[1]) *
            (1-self.weight_xyz[2])

        ])

    cdef void _generate_index_vector(self):
        """Generate vector of eight superindices containing a bead.
        
        Notes
        -----
        This is a cleaner implementation, but s about 70% slower:
        
        cdef int j
        cdef long ind_x, ind_y, ind_z
        for j in range(8):
            ind_x = ((j % 2) + self.index_xyz[0]) % self.nx
            ind_y = (((int(floor(j/2))) % 2) + self.index_xyz[1]) % self.ny
            ind_z = (((int(floor(j/4))) % 2) + self.index_xyz[2]) % self.nz
            self.nbr_inds[j] = inds_to_super_ind(
                ind_x, ind_y, ind_z, self.nx, self.ny
            )
        """
        # lower_x, lower_y, lower_z
        self.nbr_inds[0] = self.inds_xyz_to_super[
            (self.index_xyz[0]), (self.index_xyz[1]), (self.index_xyz[2])
        ]

        # upper_x, lower_y, lower_z
        self.nbr_inds[1] = self.inds_xyz_to_super[
            (1+self.index_xyz[0]), (self.index_xyz[1]), (self.index_xyz[2])
        ]

        # lower_x, upper_y, lower_z
        self.nbr_inds[2] = self.inds_xyz_to_super[
            (self.index_xyz[0]), (1+self.index_xyz[1]), (self.index_xyz[2])
        ]

        # upper_x, upper_y, lower_z
        self.nbr_inds[3] = self.inds_xyz_to_super[
            (1+self.index_xyz[0]), (1+self.index_xyz[1]), (self.index_xyz[2])
        ]

        # lower_x, lower_y, upper_z
        self.nbr_inds[4] = self.inds_xyz_to_super[
            (self.index_xyz[0]), (self.index_xyz[1]), (1+self.index_xyz[2])
        ]

        # upper_x, lower_y, upper_z
        self.nbr_inds[5] = self.inds_xyz_to_super[
            (1+self.index_xyz[0]), (self.index_xyz[1]), (1+self.index_xyz[2])
        ]

        # lower_x, upper_y, upper_z
        self.nbr_inds[6] = self.inds_xyz_to_super[
            (self.index_xyz[0]), (1+self.index_xyz[1]), (1+self.index_xyz[2])
        ]

        # upper_x, upper_y, upper_z
        self.nbr_inds[7] = self.inds_xyz_to_super[
            (1+self.index_xyz[0]), (1+self.index_xyz[1]), (1+self.index_xyz[2])
        ]

    cdef double get_E_binders_and_beads(
        self, poly.PolymerBase poly, long[:] inds, long n_inds
    ):
        """Get total energy of polymer associated with configuration and binders.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer for which densities are calculated.
        inds : long[:] by reference
            Array of bead indices for the entire polymer (to avoid re-computing
            this).
        n_inds : long
            Number of beads in the polymer.

        Returns
        -------
        double
            Field energy associated with the polymer
        """
        cdef double E_binders_beads, tot_density
        cdef long i, j, n_double_bound
        cdef dict binder_info

        # Count num. nucleosomes bound by reader proteins at both histone tails
        self.count_doubly_bound(poly, inds, n_inds, trial=0, state_change=0)

        E_binders_beads = 0
        for i, binder_info in enumerate(self.binder_dict):
            # Calculate the total density
            tot_density = 0
            for j in range(self.n_bins):
                tot_density += self.density[j, i+1]**2
            # Oligomerization
            E_binders_beads +=\
                binder_info['field_energy_prefactor'] * tot_density
            # Intranucleosome interaction
            n_double_bound = self.doubly_bound[i]
            E_binders_beads +=\
                binder_info['interaction_energy_intranucleosome'] *\
                n_double_bound

        # Nonspecific bead interaction energy
        E_binders_beads += self.nonspecific_interact_E(poly)

        return E_binders_beads

    cpdef double nonspecific_interact_E(self, poly.PolymerBase poly):
        """Get nonspecific interaction energy for the full polymer.

        Notes
        -----
        This method assumes that all beads are of the same volume.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer affected by the current MC move

        Returns
        -------
        double
            Absolute energy associated with the nonspecific interactions
            between segments of the polymer chain
        """
        cdef double bead_V, nonspecific_E, access_vol
        cdef double[:] vol_fracs

        bead_V = poly.beads[0].vol
        vol_fracs = self.get_volume_fractions(bead_V)
        nonspecific_E = 0
        for i in range(self.n_bins):
            access_vol = self.access_vols[i]
            if round(vol_fracs[i], 2) > self.vf_limit:
                nonspecific_E += E_HUGE * vol_fracs[i]
            else:
                nonspecific_E += self.chi * (access_vol / bead_V) *\
                                 vol_fracs[i] * (1 - vol_fracs[i])
        return nonspecific_E

    cdef double[:] get_volume_fractions(self, double bead_V):
        """Calculate all volume fractions of beads in bins.
        
        Notes
        -----
        We can obtain a count of beads in a voxel from the density of the voxel
        multiplied by accessible volume. Using this count of beads in the voxel
        and the volume of each bead, we can get the total volume of the beads
        in the voxel. Then, dividing the total volume of the beads by the total
        volume of the voxel gives us the volume fraction of the beads in the
        voxel. This is equivalent to multiplying the density of the beads by the
        volume of each bead.

        Parameters
        ----------
        bead_V : double
            Volume of an individual bead

        Returns
        -------
        double[:]
            Array of bead volume fractions for each bin
        """
        cdef long j
        cdef double[:] vol_fracs = np.empty((self.n_bins,), dtype=float)
        for j in range(self.n_bins):
            vol_fracs[j] = self.density[j, 0] * bead_V
        return vol_fracs

    cdef double[:, ::1] get_coordinates_at_inds(
        self, double[:, ::1]& r, long[:]& inds, long n_inds
    ):
        """
        Get position of specified beads from current polymer configuration.

        Parameters
        ----------
        r : array_like (N, 3) of double by reference
            x, y, z coordinates of every bead in the polymer where rows
            coorespond to individual beads and columns correspond to 
            Cartesian (x, y, z) coordinates
        inds : array_like (M,) of long by reference
            Indices of M beads involved in the move, where `M=n_inds`
        n_inds : long
            Number of bead indices affected by a move

        Returns
        -------
        double[:, ::1]
            Coordinates for specified indicies from current polymer
            configuration where rows coorespond to individual beads and
            columns correspond to Cartesian (x, y, z) coordinates
        """
        cdef long i, j
        cdef double[:, ::1] r_new
        r_new = np.empty((n_inds, 3), dtype=float)
        for i in range(n_inds):
            for j in range(3):
                r_new[i, j] = r[inds[i], j]
        return r_new

    cdef long[:, ::1] get_states_at_inds(
        self, poly.PolymerBase poly, long[:] inds, long n_inds
    ):
        """Get binding states of specified beads from current polymer config.

        Parameters
        ----------
        poly : poly,PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N, ) of long
            Indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move

        Returns
        -------
        long[:, ::1]
            Binding states on polymer at specified bead indices, where rows
            identify individual beads and columns correspond to binding states
            for each reader protein
        """
        cdef long i, j
        cdef long[:, ::1] states
        states = np.empty((n_inds, poly.num_binders), dtype=int)
        for i in range(n_inds):
            for j in range(poly.num_binders):    
                states[i, j] = poly.states[inds[i], j]
        return states


cdef long inds_to_super_ind(
    long ind_x, long ind_y, long ind_z, long nx, long ny
):
    """Determine the super index of a voxel based on three dimensional indices.

    Notes
    -----
    Given the x, y, and z voxel indices and the number of voxels in the x and y
    dimensions, calculate the super-index of a bin.

    Parameters
    ----------
    ind_x, ind_y, ind_z : long
        Voxel index positions in the x, y, and z dimensions
    nx, ny : long
        Number of bins in the x and y directions of Cartesian space

    Returns
    -------
    long
        Super-index position of the voxel
    """
    cdef long super_ind = ind_x + ind_y * nx + ind_z * nx * ny
    return super_ind


cpdef long[:] super_ind_to_inds(long super_ind, long nx, long ny):
    """Calculate the three-dimensional indices from a super-index.
    
    Notes
    -----
    Given the super-index of a voxel and the number of voxels in the x and y
    directions, calculate the three-dimensional indices of the voxel.

    Parameters
    ----------
    super_ind : long
        Super-index of a voxel
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction

    Returns
    -------
    long[:]
        Indices of bin in x, y, and z directions
    """
    cdef long[:] xyz_inds = np.zeros(3, dtype=int)
    xyz_inds[2] = np.floor(super_ind / (nx * ny))
    xyz_inds[1] = np.floor((super_ind - xyz_inds[2] * nx * ny) / nx)
    xyz_inds[0] = super_ind - (xyz_inds[2] * (nx * ny) + xyz_inds[1] * nx)
    return xyz_inds


cpdef dict assign_beads_to_bins(
    double[:, ::1] r_poly, long n_inds, long nx, long ny, long nz,
    double x_width, double y_width, double z_width
):
    """Create a mapping from bin indices to list of associated beads.
    
    Parameters
    ----------
    r_poly : double[:, ::1]
        Array of bead positions
    n_inds : long
        Number of beads - must be equivalent in length to r_poly
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction
    nz : long
        Number of bins in the z direction
    x_width : double
        Widths of space in the x direction
    y_width : double
        Widths of space in the y direction
    z_width : double
        Widths of space in the z direction

    Returns
    -------
    dict
        Mapping of bin indices to associated beads, where keys indicate
        the bin indices and values are lists of beads contained in the bin
    """
    cdef long i, index_x, index_y, index_z, n_bins
    cdef dict bin_map = {}
    cdef double[:] r_i, x_poly_box, y_poly_box, z_poly_box
    cdef double x_poly_box_minus, y_poly_box_minus, z_poly_box_minus
    cdef double dx, dy, dz

    x_poly_box = r_poly[:, 0].copy()
    y_poly_box = r_poly[:, 1].copy()
    z_poly_box = r_poly[:, 2].copy()

    dx = x_width / nx
    dy = y_width / ny
    dz = z_width / nz

    n_bins = nx * ny * nz

    for i in range(n_bins):
        bin_map[i] = []

    for i in range(n_inds):
        x_poly_box_minus = floor((r_poly[i, 0] - 0.5 * dx) / x_width)
        x_poly_box[i] -= 0.5 * dx + x_width * x_poly_box_minus
        y_poly_box_minus = floor((r_poly[i, 1] - 0.5 * dy) / y_width)
        y_poly_box[i] -= 0.5 * dy + y_width * y_poly_box_minus
        z_poly_box_minus = floor((r_poly[i, 2] - 0.5 * dz) / z_width)
        z_poly_box[i] -= 0.5 * dz + z_width * z_poly_box_minus

    for i in range(n_inds):
        index_x = <long>floor(nx * x_poly_box[i] / x_width)
        index_y = <long>floor(ny * y_poly_box[i] / y_width)
        index_z = <long>floor(nz * z_poly_box[i] / z_width)
        index_x0y0z0 = index_x + nx * index_y + nx * ny * index_z

        bin_map[index_x0y0z0].append(i)

    return bin_map


cpdef dict get_neighboring_bins(long nx, long ny, long nz):
    """Generate map of bin indices to all immediately neighboring bin indices.

    Parameters
    ----------
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction
    nz : long
        Number of bins in the z direction

    Returns
    -------
    Dict[long, np.ndarray(ndims=1, dtype=long)]
        Dictionary where keys indicate bin indices and values provide a list of
        all immediately neighboring bin indices, including the key.
    """
    cdef long num_bins, i
    cdef dict neighbors

    num_bins = nx * ny * nz
    neighbors = {}
    for i in range(num_bins):
        neighbors[i] = get_neighbors_at_ind(nx, ny, nz, i, num_bins)
    return neighbors


cpdef long[:] get_neighbors_at_ind(
    long nx, long ny, long nz, long ind, long num_bins
):
    """Identify the bin indices immediately neighboring a specific bin index.

    Parameters
    ----------
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction
    nz : long
        Number of bins in the z direction
    ind : long
        Super-index of bin for which neighbors are desired
    num_bins : long
        Number of bins contained  in the field
    
    Returns
    -------
    np.ndarray(ndims=1, dtype=long)
        Array of nine bin super-indices immediately neighboring and containing
        the bin specified by `ind`
    """
    cdef long[:] nbr_bins_flat = np.zeros(27, dtype=int)
    xyz = super_ind_to_inds(ind, nx, ny)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    nbr_bins = np.zeros((3, 3, 3), dtype=int)
    for i, dx in enumerate(range(-1, 2)):
        for j, dy in enumerate(range(-1, 2)):
            for k, dz in enumerate(range(-1, 2)):
                nbr_bins[i, j, k] = inds_to_super_ind(
                    (x + dx) % nx, (y + dy) % ny, (z + dz) % nz, nx, ny
                )
    nbr_bins_flat = nbr_bins.flatten()
    return nbr_bins_flat


cpdef dict get_blocks(long num_beads, long block_size):
    """Coarse grain individual beads into blocks of linearly adjacent beads.
    
    Notes
    -----
    Used to improve computational tractibility of contact map generation.

    Parameters
    ----------
    num_beads : long
        Number of beads in full-resolution polymer
    block_size : long
        Number of beads to place into a single block in course-grained model

    Return
    ------
    dict[long, long]
        Mapping of each bead ID to its block.
    """
    cdef long i
    cdef dict blocks

    blocks = {}
    for i in range(num_beads):
        blocks[i] = int(floor(i / block_size))

    return blocks
