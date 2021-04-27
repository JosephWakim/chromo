"""
Fields discretize space to efficiently track changes in Mark energy.

Creates a field object that contains parameters for the field calculations
and functions to generate the densities
"""
import numpy as np
import pandas as pd

from .util import combine_repeat
from .mc.moves import MCAdapter


class FieldBase:
    """
    A discretization of space for computing energies.

    Must be subclassed to be useful.
    """

    _field_descriptors = []

    @property
    def name(self):
        """For now, there's only one field per sim, so classname works."""
        return self.__class__.__name__

    def __init__(self):
        """Construct a field holding no Polymers, tracking no Marks."""
        self.polymers = []
        self.marks = pd.DataFrame()

    def __str__(self):
        """Print representation of empty field."""
        return "Field<>"

    def __contains__(self, poly):
        """Check if a polymer is currently set to interact with this field."""
        # No polymers interact with the base field object
        return self.polymers.__contains__(poly)

    def compute_dE(self, poly, inds, r, t3, t2, states):
        """
        Compute the change in field energy due to a proposed move.

        In order to prevent unnecessary computation, you can omit any
        parameters which where completely unchanged by the move.

        Parameters
        ----------
        poly : `chromo.Polymer`
            The polymer which has been moved.
        inds : array_like (N,)
            Indices of beads being moved
        r : (N, 3) array_like of float, optional
            The proposed new positions of the moved beads. Throughout this
            function, ``N = indf - ind0 + 1`` is the number of moved beads.
        t3 : (N, 3) array_like of float, optional
            The proposed new tangent vectors.
        t2 : (N, 3) array_like of float, optional
            Proposed new material normals.
        states : (N, M) array_like of int, optional
            The proposed new chemical states, where *M* is the number of
            chemical states associated with the given polymer.

        Returns
        -------
        float
            The change in energy caused by the Polymer's movement in this
            field.

        """
        pass


class Reconstructor:
    """
    Defer construction of `Field` until after `Polymer`/`Mark` instances.

    Constructs a kwargs object that can be re-passed to the appropriate `Field`
    constructor when they become available.
    """

    def __init__(self, cls, **kwargs):
        """Construct our Reconstructor."""
        self.field_constructor = cls
        self.kwargs = kwargs

    def finalize(self, polymers, marks):
        """Finish construction of appropriate `Field` object."""
        return self.field_constructor(polymers=polymers, marks=marks,
                                      **self.kwargs)

    @classmethod
    def from_file(cls, path):
        """Assume class name is encoded in file name."""
        constructor = globals()[path.name]
        # read info as a series
        kwargs = pd.read_csv(path).iloc[0].to_dict()
        return cls(constructor, **kwargs)

    def __call__(self, polymers, marks):
        """Synonym for `Reconstructor.finalize`."""
        return self.finalize(polymers, marks)


class UniformDensityField(FieldBase):
    """
    Rectilinear discretization of a rectangular box.

    Computes field energies at the corners of each box. The bead in each box
    contributes "mass" to each vertex linearly based on its position in the
    box. This is much more stable numerically than just using the boxes as
    "bins" as would be done in a e.g. finite-differences discretization.
    """

    _field_descriptors = ['x_width', 'nx', 'y_width', 'ny', 'z_width', 'nz']

    def __init__(self, polymers, marks, x_width, nx, y_width, ny, z_width,
                 nz):
        """
        Construct a UniformDensityField.

        Parameters
        ----------
        polymers : Sequence[Polymer]
            Name of (or object representing) each polymer in the field.
        marks : pd.DataFrame
            Output of `chromo.marks.make_mark_collection` for the `Mark`s to be
            used.
        x_width : float
            Width of the box containing the field in the x-direction.
        nx : int
            Number of bins in the x-direction.
        y_width : float
            Width of the box containing the field in the y-direction.
        ny : int
            Number of bins in the y-direction.
        z_width : float
            Width of the box containing the field in the z-direction.
        nz : int
            Number of bins in the z-direction.

        """
        self.polymers = polymers
        self.marks = marks
        for poly in polymers:
            if poly.num_marks != len(marks):
                raise NotImplementedError("For now, all polymers must use all"
                                          " of the same marks.")
        self.x_width = x_width
        self.y_width = y_width
        self.z_width = z_width
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = x_width / nx
        self.dy = y_width / ny
        self.dz = z_width / nz
        self.n_bins = nx * ny * nz
        self.vol_bin = x_width * y_width * z_width / self.n_bins
        self.bin_index = UniformDensityField._get_corner_bin_index(
                self.nx, self.ny, self.nz)
        # one column of density for each state and one for the actual bead
        # density of the polymer
        self.num_marks = len(marks)
        self.density = np.zeros((self.n_bins, self.num_marks + 1), 'd')
        self._recompute_field()  # initialize the density field

    def to_file(self, path):
        """Save Field description and Polymer/Mark names to CSV as Series."""
        rows = {name: self.__dict__[name] for name in self._field_descriptors}
        for i, polymer in enumerate(self.polymers):
            rows[polymer.name] = 'polymer'
        for i, mark in self.marks.iterrows():
            # careful! mark.name is the Series.name attribute
            rows[mark['name']] = 'mark'
        # prints just key,value for each key in rows
        return pd.Series(rows).to_csv(path, header=False)

    @classmethod
    def from_file(cls, path, polymers, marks):
        """Recover field saved with `.to_file`. Requires Poly/Mark inputs."""
        # the 0th column will be the index, the 1th column holds the data...
        field_series = pd.read_csv(path, header=None, index_col=0)[1]
        kwargs = pd.to_numeric(field_series[cls._field_descriptors]).to_dict()
        polymer_names = field_series[field_series == 'polymer'].index.values
        mark_names = field_series[field_series == 'mark'].index.values

        err_prefix = f"Tried to instantiate class:{cls} from file:{path} with "
        if len(polymers) != len(polymer_names):
            raise ValueError(err_prefix + f"{len(polymers)} polymers, but "
                             f" there are {len(polymer_names)} listed.")
        for polymer in polymers:
            if polymer.name not in polymer_names:
                raise ValueError(err_prefix + f"polymer:{polymer.name}, but "
                                 " this polymer was not present in file.")
        if len(marks) != len(mark_names):
            raise ValueError(err_prefix + f"{len(marks)} marks, but "
                             f" there are {len(mark_names)} listed.")
        for i, mark in marks.iterrows():
            if mark['name'] not in mark_names:
                raise ValueError(err_prefix + f"mark:{mark}, but "
                                 " this mark was not present in file.")

        return cls(polymers=polymers, marks=marks, **kwargs)

    @staticmethod
    def _get_corner_bin_index(nx, ny, nz):
        """
        Set up the index array for density bins corners.

        TODO: this should just be a higher-dimensional array to avoid having to
        do any of this math.
        """
        bin_index = np.zeros((nx * ny * nz, 8), 'd')
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
                    bin_index[count, :] = [
                        index_x + nx * index_y + nx * ny * index_z,
                        index_xp1 + nx * index_y + nx * ny * index_z,
                        index_x + nx * index_yp1 + nx * ny * index_z,
                        index_xp1 + nx * index_yp1 + nx * ny * index_z,
                        index_x + nx * index_y + nx * ny * index_zp1,
                        index_xp1 + nx * index_y + nx * ny * index_zp1,
                        index_x + nx * index_yp1 + nx * ny * index_zp1,
                        index_xp1 + nx * index_yp1 + nx * ny * index_zp1]
                    count += 1
        return bin_index

    def __str__(self):
        """Print summary of the UniformDensityField."""
        n_poly = len(self.polymers)
        return f"UniformDensityField<nx={self.nx},ny={self.ny},nz={self.nz}," \
               f"npoly={n_poly}>"

    def _recompute_field(self, check_consistency=False):
        """
        Completely recompute the energy from scratch.

        Re-running this not in __init__ and using *check_consistency* to verify
        that the answer is not too far from the answer that has been built up
        over monte carlo time is one of the best ways to test for consistency
        of the code.
        """
        new_density = self.density.copy()
        for i, poly in enumerate(self.polymers):
            density_poly, index_xyz = self._calc_density(
                    poly.r, poly.states, 0, poly.num_beads)
            new_density[index_xyz, :] += density_poly
        if check_consistency:
            if not np.all(np.isclose(new_density, self.density)):
                raise RuntimeError(f"Recomputing energy of {self} from scratch"
                                   " produced inconsistent results. There is a"
                                   " bug in the code.")

    def compute_dE(self, poly, inds, r_trial, t3_trial, t2_trial, states_trial, 
        continuous_inds):
        """
        Compute change in energy based on proposed new polymer location.

        Requires the polymer to be moved in order to compare to old state.

        Parameters
        ----------
        inds : array_like (N, 3)
            Ordered indices of N beads involved in the move
        r_trial : array_like (N, 3)
            Array of position vectors for N beads involved in the move
        t3_trial : array_like (N, 3)
            Array of t3 tangent vectors for N beads involved in the move
        t2_trial : array_like (N, 3)
            Array of t2 tangent vectors for N beads involved in the move
        states_trial : array_like (N, M)
            Array of M bead states for N beads involved in the move
        continuous_inds : bool
            Flag indicating whether moves affects a continuous region or not

        Returns
        -------
        dE_marks : float
            Change in energy assocaited with the polumer configuration
        """
        # for now, do not actually use the fact that some moves do not affect
        # certain parts of the polymer for any optimization. Fill in any `None`
        # values with the old state of the polymer explicitly.
        r, t3, t2, states = MCAdapter.replace_none(
            poly,
            inds,
            r_trial,
            t3_trial,
            t2_trial,
            states_trial,
            continuous_inds
        )
        r = np.atleast_2d(r)
        t3 = np.atleast_2d(t3)
        t2 = np.atleast_2d(t2)

        density_poly, index_xyz = self._calc_density(
            poly.r[inds, :], poly.states[inds, :], inds[0], inds[-1]+1)
        density_poly_trial, index_xyz_trial = self._calc_density(
            r, states, inds[0], inds[-1]+1)
        delta_density_poly_total = \
            np.concatenate((density_poly_trial, -density_poly))
        delta_index_xyz_total = \
            np.concatenate((index_xyz_trial, index_xyz)).astype(int)
        delta_density, delta_index_xyz = \
            combine_repeat(delta_density_poly_total, delta_index_xyz_total)

        dE_marks = 0
        for i, (_, mark_info) in enumerate(self.marks.iterrows()):
            dE_marks += 0.5 * mark_info.interaction_energy * np.sum(
                (
                    delta_density[:, i + 1]
                    + self.density[delta_index_xyz, i + 1]
                 ) ** 2
                - self.density[delta_index_xyz, i + 1] ** 2
            )
        return dE_marks

    def _find_bins(self, r_poly):
        # Find the (0,0,0) bins for the beads and the associated weights
        x_poly_box = (r_poly[:, 0] - 0.5 * self.dx - self.x_width * np.floor(
            (r_poly[:, 0] - 0.5 * self.dx) / self.x_width))
        index_x = np.floor(self.nx * x_poly_box / self.x_width).astype(int)
        weight_x = 1 - (x_poly_box / self.dx - index_x)

        y_poly_box = (r_poly[:, 1] - 0.5 * self.dy - self.y_width * np.floor(
            (r_poly[:, 1] - 0.5 * self.dy) / self.y_width))
        index_y = np.floor(self.ny * y_poly_box / self.y_width).astype(int)
        weight_y = 1 - (y_poly_box / self.dy - index_y)

        z_poly_box = (r_poly[:, 2] - 0.5 * self.dz - self.z_width * np.floor(
            (r_poly[:, 2] - 0.5 * self.dz) / self.z_width))
        index_z = np.floor(self.nz * z_poly_box / self.z_width).astype(int)
        weight_z = 1 - (z_poly_box / self.dz - index_z)

        index_x0y0z0 = index_x + self.nx*index_y + self.nx*self.ny*index_z

        return weight_x, weight_y, weight_z, index_x0y0z0

    def _generate_weight_vector(self, weight_x, weight_y, weight_z):
        """Generate weight array for all of the bins containing the beads."""
        weight = np.array([weight_x * weight_y * weight_z,
                           (1 - weight_x) * weight_y * weight_z,
                           weight_x * (1 - weight_y) * weight_z,
                           (1 - weight_x) * (1 - weight_y) * weight_z,
                           weight_x * weight_y * (1 - weight_z),
                           (1 - weight_x) * weight_y * (1 - weight_z),
                           weight_x * (1 - weight_y) * (1 - weight_z),
                           (1 - weight_x) * (1 - weight_y) * (1 - weight_z)]
                          ).flatten()
        return weight

    def _generate_index_vector(self, index_x0y0z0):

        # TODO: this can be done in one line
        index_xyz_total = index_x0y0z0
        index_xyz_total = np.concatenate(
            (index_xyz_total, self.bin_index[index_x0y0z0, 1])
        ).astype(int)
        index_xyz_total = np.concatenate(
            (index_xyz_total, self.bin_index[index_x0y0z0, 2])
        ).astype(int)
        index_xyz_total = np.concatenate(
            (index_xyz_total, self.bin_index[index_x0y0z0, 3])
        ).astype(int)
        index_xyz_total = np.concatenate(
            (index_xyz_total, self.bin_index[index_x0y0z0, 4])
        ).astype(int)
        index_xyz_total = np.concatenate(
            (index_xyz_total, self.bin_index[index_x0y0z0, 5])
        ).astype(int)
        index_xyz_total = np.concatenate(
            (index_xyz_total, self.bin_index[index_x0y0z0, 6])
        ).astype(int)
        index_xyz_total = np.concatenate(
            (index_xyz_total, self.bin_index[index_x0y0z0, 7])
        ).astype(int)
        return index_xyz_total

    def _generate_weights_and_indices(self, r_poly):
        """ Generate weight and indices vector from nucleosome coordiantes."""

        # Find the (0,0,0) bins for the beads and the associated weights
        weight_x, weight_y, weight_z, index_x0y0z0 = self._find_bins(r_poly)
        # Generate the weight array for all of the bins containing the beads
        weight = self._generate_weight_vector(weight_x, weight_y, weight_z)
        # Generate index vector
        index_xyz_total = self._generate_index_vector(index_x0y0z0)

        return weight, index_xyz_total

    def _calc_density(self, r_poly, states, ind0, indf):

        states = np.atleast_2d(states)
        num_marks = states.shape[1]
        num_beads = r_poly.shape[0]

        # Generate weights and indices
        weight, index_xyz_total = self._generate_weights_and_indices(r_poly)

        # Initialize density vector
        density_total = np.zeros((len(weight), num_marks + 1), 'd')

        density_total[:, 0] = weight
        for ind_mark in range(num_marks):
            for ind_corner in range(8):
                ind_corner0 = ind_corner * (num_beads)
                ind_cornerf = ind_corner0 + num_beads
                density_total[ind_corner0:ind_cornerf, ind_mark + 1] = \
                    weight[ind_corner0:ind_cornerf] \
                    * states[:, ind_mark]

        # Combine repeat entries in the density array
        density, index_xyz = combine_repeat(density_total, index_xyz_total)

        return density, index_xyz
