"""Components that will make up our simulation.

Various types of polymers, solvents, and other simulation components should be
defined here.
"""
# Built-in Modules
from abc import ABC, abstractmethod
from typing import (Callable, Sequence, Optional, Dict, List)

# External Modules
import numpy as np
import pandas as pd

# Custom Modules
import chromo.beads as beads
import chromo.util.mc_stat as mc_stat
from .util import dss_params
from .util import poly_paths as paths


class Polymer(ABC):
    """Abstract class representation of an arbitrary polymer.

    Specifies the attributes and methods required to define a new polymer class
    and implements utility methods common to all polymer classes.
    """

    required_attrs = [
        "name", "r", "t3", "t2", "states", "mark_names", "num_marks",
        "beads", "num_beads", "lp"
    ]

    _arrays = 'r', 't3', 't2', 'states'     # arrays saved to file
    _3d_arrays = 'r', 't3', 't2'    # arrays w/ multi-indexed values (x, y, z)

    def __init__(
        self,
        name: str,
        r: np.ndarray,
        *,
        t3: Optional[np.ndarray] = None,
        t2: Optional[np.ndarray] = None,
        states: Optional[np.ndarray] = None,
        mark_names: Optional[Sequence[str]] = None,
        log_path: Optional[str] = None
    ):
        """Any polymer class requires these key attributes.

        Specifying these attributes ensures that new polymer classes are
        compatible with the rest of the codebase.

        Optional arguments must be specified by name.

        Parameters
        ----------
        name : str
            A name for convenient repr. Should be a valid filename.
        r : (N, 3) array_like of float
            The positions of each bead.
        t3 : Optional (N, 3) array_like of float
            The tangent vector to each bead in global coordinates. Default is
            None. If None, t3 bead orientations will not be tracked.
        t2 : Optional (N, 3) array_like of float
            A material normal to each bead in global coordinates. Default is
            None. If None, t2 bead orientations will not be tracked.
        states : Optional (N, M) array_like of int
            State of each of the M epigenetic marks being tracked for each
            bead. Default is None. If None, beads will be unbound by any marks.
        mark_names : Optional (M, ) str or Sequence[str]
            The name of each chemical modification tracked in `states`, for
            each of tracking which mark is which. Default is None. Number of
            `mark_names` specified must match number of columns in `states`.
        log_path : Optional str
            Path to the configuration tracker log. Default is None. If None, no
            configuration tracker will be initialized.
        """
        self.name = name
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.states = states
        self.mark_names = mark_names
        self.num_marks = self.get_num_marks()
        self.beads = self.construct_beads()
        self.num_beads = self.get_num_beads()
        self.log_path = log_path
        self.update_log_path(log_path)
        if states is not None:
            self.check_marks(states, mark_names)
        self.lp = None

    @abstractmethod
    def compute_dE(self):
        """Compute the change in configurational energy of a polymer.
        """
        pass

    @abstractmethod
    def construct_beads(self) -> Dict[int, beads.Bead]:
        """Construct a list of beads forming the polymer.

        Returns
        -------
        Dict[int, beads.Bead]
            Dictionary of beads constructing the polymer chain. Keys represent
            bead IDs, and values represent beads.Bead objects.
        """
        pass

    def __str__(self):
        """Return string representation of the Polymer.
        """
        return f"Polymer<{self.name}, nbeads={self.num_beads}, " \
               f"nmarks={self.num_marks}>"

    def check_attrs(self):
        """Check that the required attributes are specified in the class.
        """
        for attr in self.required_attrs:
            if not hasattr(self, attr):
                raise NotImplementedError(
                    "Polymer subclass missing required attribute: " + attr
                )
        if self.lp is None:
            raise ValueError(
                "Specify the persistence length in the subclass of Polymer"
            )

    def check_marks(self, states, mark_names):
        """Verify that specified mark states and names are valid.

        Parameters
        ----------
        states : (N, M) array_like of int
            State of each of the M epigenetic marks being tracked for each
            bead.
        mark_names : (M, ) str or Sequence[str]
            The name of each chemical modification tracked in `states`, for
            each of tracking which mark is which.
        """
        self.states = self.states.reshape(states.shape[0], -1)
        num_beads, num_marks = self.states.shape
        if num_marks != len(mark_names):
            raise ValueError("Each chemical state must be given a name.")
        if num_beads != len(self.r):
            raise ValueError("Initial epigenetic state of wrong length.")

    def update_log_path(self, log_path: Optional[str]):
        """Update the path to the configuration tracker log.

        Parameters
        ----------
        log_path : str
            Path to the configuration tracker log. If None, no configuration
            tracker will be initialized. (default = None)
        """
        if log_path is not None:
            self.configuration_tracker = \
                mc_stat.ConfigurationTracker(log_path, self.r)

    def fill_in_None_Parameters(self, inds, r, t3, t2, states):
        """Substitute polymer parameters when attribute is None type.
        """
        if r is None:
            r = self.r[inds, :]
        if t3 is None:
            t3 = self.t3[inds, :]
        if t2 is None:
            t2 = self.t2[inds, :]
        if states is None:
            states = self.states[inds, :]
        return r, t3, t2, states

    def get_prop(self, inds: List[int], prop: str):
        """Get specified property of beads at listed indices.

        Parameters
        ----------
        inds : List[int] or (M, ) array_like
            Indices at which to isolate position vectors

        Returns
        -------
        np.ndarray (M, 3)
            Positions of beads at specified indices.
        """
        inds = np.atleast_1d(inds)
        return np.array(
            [self.beads[i].__dict__[prop] for i in inds]
        )

    def get_all(self, prop) -> Optional[np.ndarray]:
        """Get some bead property value from all beads

        Parameters
        ----------
        prop : str
            Name of the property to obtain from all beads

        Returns
        -------
        Optional[np.ndarray]
            Data frame of property values from all beads (default = None)
        """
        if self.beads[0].__dict__[prop] is not None:
            array = np.array(
                [self.beads[i].__dict__[prop] for i in range(len(self.beads))]
            )
            if len(array.shape) == 1:
                array = np.atleast_2d(array).T
            return array
        return None

    def to_dataframe(self):
        """Write canonical CSV representation of the Polymer to file.

        The top-level multiindex values for the columns should correspond to
        the kwargs, to simplify unpacking the structure (and to make for easy
        accessing of the dataframe.

        First get a listing of all the parts of the polymer that have to be
        saved. These are defined in the class's `_arrays` attribute.

        Next, separate out the arrays into two types: "vector" arrays are
        composed of multiple columns (like r) and so we need to build a multi-
        index for putting them into the data frame correctly.

        All other arrays can be added as a single column to the data frame.

        Epigenmark names is a special case because it does not fit with with
        dimensions of the other properties, which are saved per bead.

        Construct the parts of the DataFrame that need a multi-index.

        TODO: remove "list" after numpy fixes issue 17305:
        https://github.com/numpy/numpy/issues/17305

        Replace:
            `vector_arr = np.concatenate(list(vector_arrs.values()), axis=1)`
        with:
            `vector_arr = np.concatenate(vector_arrs.values(), axis=1)`

        After adding multi-index properties, add remaining arrays one-by-one.
        """
        arrays = {name: self.__dict__[name] for name in self._arrays
                  if self.__dict__[name] is not None}
        vector_arrs = {}
        regular_arrs = {}

        for name, arr in arrays.items():
            if name in self._3d_arrays:
                vector_arrs[name] = arr
            elif name != 'states':
                regular_arrs[name] = arr
        vector_arr = np.concatenate(list(vector_arrs.values()), axis=1)
        vector_index = pd.MultiIndex.from_product(
            [vector_arrs.keys(), ('x', 'y', 'z')]
        )
        vector_df = pd.DataFrame(vector_arr, columns=vector_index)
        states_index = pd.MultiIndex.from_tuples(
                [('states', name) for name in self.mark_names])
        states_df = pd.DataFrame(self.states, columns=states_index)
        df = pd.concat([vector_df, states_df], axis=1)

        for name, arr in regular_arrs.items():
            df[name] = arr
        return df

    def to_csv(self, path):
        """Save Polymer object to CSV file as DataFrame.
        """
        return self.to_dataframe().to_csv(path)

    def to_file(self, path):
        """Synonym for `to_csv` to conform to `make_reproducible` spec.
        """
        return self.to_csv(path)

    @classmethod
    def from_csv(cls, csv_file: str) -> pd.DataFrame:
        """Construct Polymer from CSV file.

        Parameters
        ----------
        csv_file : str
            Path to CSV file from which to construct polymer

        Returns
        -------
        Polymer
            Object representation of a polymer
        """
        df = pd.read_csv(csv_file)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df, name=None):
        """Construct Polymer object from DataFrame. Inverts `.to_dataframe`.
        """
        # top-level multiindex values correspond to kwargs
        kwnames = np.unique(df.columns.get_level_values(0))
        kwargs = {name: df[name].to_numpy() for name in kwnames}
        # extract names of each epigenetic state from multi-index
        if 'states' in df:
            mark_names = df['states'].columns.to_numpy()
            kwargs['mark_names'] = mark_names
        return cls(name, **kwargs)

    @classmethod
    def from_file(cls, path, name=None):
        """Construct Polymer object from string representation.
        """
        if name is None:
            name = path.name
        return cls.from_dataframe(
            pd.read_csv(path, header=[0, 1], index_col=0),
            name
        )

    def get_num_marks(self):
        """Return number of states tracked per bead.
        """
        if len(self.states.shape) > 1:
            return self.states.shape[1]
        else:
            return 1

    def get_num_beads(self):
        """Return number of beads in the polymer.
        """
        return self.r.shape[0]


class Chromatin(Polymer):
    """The positions and chemical state of a discrete polymer.

    The polymer carries around a set of coordinates `r` of shape
    `(num_beads, 3)`, a triad of material normals `t_i` for `i` in
    `{1,2,3}`, and some number of chemical states per bead.

    Polymer also carries around a dictionary of beads of length `num_beads`.
    Each bead has a set of coordiantes `r` of length (3, 1). The beads also
    have a triad of material normals `t_i` for `i` in `{1,2,3}`. The beads
    optionally carry around some number of chemical states.

    Its material properties are completely defined by these positions, chemical
    states, and the length of polymer (in Kuhn lengths) simulated by each bead.

    Since this codebase is primarily used to simulate DNA, information about
    the chemical properties of each epigenetic mark are stored in `Epigenmark`
    objects.

    TODO: allow differential discretization, and decay to constant
    discretization naturally.
    """

    _arrays = 'r', 't3', 't2', 'states', 'bead_length'  # arrays saved to file
    _3d_arrays = 'r', 't3', 't2'    # arrays w/ multi-indexed values (x, y, z)

    def __init__(
        self,
        name: str,
        r: np.ndarray,
        *,
        bead_length: float,
        t3: Optional[np.ndarray] = None,
        t2: Optional[np.ndarray] = None,
        states: Optional[np.ndarray] = None,
        mark_names: Optional[Sequence[str]] = None,
        log_path: Optional[str] = None
    ):
        """Construct a `Chromatin` fiber object as a subclass of `Polymer`.

        Because the persistence length of chromatin is fixed at 53 nm, or
        53 / 0.34 base pairs, it will be hard-coded into the `Chromatin` class
        as the `lp` attribute.

        NOTE: For now, when loading polymer parameters for confirmational
        energy, the array-like of `bead_length` is ignored.

        Parameters
        ----------
        name : str
            A name for convenient repr. Should be a valid filename.
        r : (N, 3) array_like of float
            The positions of each bead.
        bead_length : float or (N,) array_like of float
            The amount of polymer path length between this bead and the next
            bead. For now, a constant value is assumed (the first value if an
            array is passed).
        t3 : (N, 3) array_like of float
            The tangent vector to each bead in global coordinates.
        t2 : (N, 3) array_like of float
            A material normal to each bead in global coordinates.
        states : (N, M) array_like of int
            State of each of the M epigenetic marks being tracked for each
            bead.
        mark_names : (M, ) str or Sequence[str]
            The name of each chemical modification tracked in `states`, for
            each of tracking which mark is which.
        log_path : Optional[str]
            Path to the configuration tracker log. If None, no configuration
            tracker will be initialized. (default = None)
        """
        bead_length = np.broadcast_to(
            np.atleast_1d(bead_length), (len(r), 1)
        )
        self.bead_length = bead_length
        super(Chromatin, self).__init__(
            name, r, t3=t3, t2=t2, states=states, mark_names=mark_names,
            log_path=log_path
        )
        # Simulation units are in bsae pairs
        self.lp = 53 / 0.34     # persistence length in base pairs
        self.delta, self.eps_bend, self.eps_par,\
            self.eps_perp, self.gamma, self.eta \
            = self._find_parameters(bead_length[0])
        self.check_attrs()

    def construct_beads(self) -> Dict[int, beads.Bead]:
        """Construct Nucleosome objects forming beads of our Chromatin polymer.

        Returns
        -------
        Dict[int, beads.Nucleosome]
            Dictionary of Nucleosome objects forming our Chroamtin fiber. Keys
            indicate nucleosome IDs, and values indicate Nucleosome objects.
        """
        return {
            i: beads.Nucleosome(
                id=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                bead_length=self.bead_length[i],
                states=self.states[i],
                mark_names=self.mark_names
            ) for i in range(len(self.r))
        }

    def compute_dE(
        self, inds, r_trial, t3_trial, t2_trial, states_trial, continuous_inds
    ):
        """Compute the change in polymer energy for move to proposed state.

        For `None` values in trial polymer states, fill in those values with
        respective quantities from the polymer prior to the proposed move.

        Initialize the change in polymer energy. If the indices affected by the
        MC move are continuous, then compute changes in polymer energy for the
        specified continuous bead range. If indices affected by the MC move are
        non-continuous (the MC move is not applied to a continuous polymer
        segment), then calculate the change in polymer energy individually at
        each affected bead.

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
        delta_energy_poly : float
            Change in polymer energy assocaited with the trial move
        """
        r_trial, t3_trial, t2_trial, states_trial = \
            self.fill_in_None_Parameters(
                inds, r_trial, t3_trial, t2_trial, states_trial
            )
        delta_energy_poly = 0
        if continuous_inds:
            ind0 = min(inds)
            indf = max(inds) + 1
            delta_energy_poly += self.continuous_dE_poly(
                ind0, indf, r_trial, t3_trial, t2_trial
            )
        else:
            for ind in inds:
                ind0 = ind
                indf = ind + 1
                delta_energy_poly += self.continuous_dE_poly(
                    ind0, indf, r_trial, t3_trial, t2_trial
                )
        return delta_energy_poly

    def continuous_dE_poly(self, ind0, indf, r_trial, t3_trial, t2_trial):
        """Compute change in polymer energy for a continuous bead region.

        The internal configuration of a continuous segment selected for a move
        is unaffected; therefore, change in polymer energy can be determined
        from the beads at the ends of the selected segment.

        If a bound of the selected segment exists at the end of the polymer,
        then that bound does not contribute to a change in polymer energy.

        For bounds of affected beads inside the polymer, begin by isolating
        the position and orientation vectors of those bounds and their
        neighbors, then calculate the change in polymer energy for the bead
        pair.

        Paramaters
        ----------
        ind0 : int
            Index of the first bead in the continuous region
        indf : int
            One past the index of the last bead in the continuous region
        r_trial : array_like (N, 3)
            Array of coordinates for bead indices in range(ind0:indf)
        t3_trial : array_like (N, 3)
            Array of t3 tangents for bead indices in range(ind0:indf)
        t2_trial : array_like (N, 3)
            Array of t2 tangents for bead indices in range(ind0:indf)

        Returns
        -------
        delta_energy_poly : float
            Change in energy of polymer associated with trial move
        """
        delta_energy_poly = 0
        if ind0 != 0:
            r_0 = self.r[ind0 - 1, :]
            r_1 = self.r[ind0, :]
            test_r_1 = r_trial[0, :]
            t3_0 = self.t3[ind0 - 1, :]
            t3_1 = self.t3[ind0, :]
            test_t3_1 = t3_trial[0, :]
            t2_0 = self.t2[ind0 - 1, :]
            t2_1 = self.t2[ind0, :]
            test_t2_1 = t2_trial[0, :]
            delta_energy_poly += self.bead_pair_dE_poly(
                r_0, r_1, test_r_1, t3_0, t3_1, test_t3_1, t2_0, t2_1,
                test_t2_1
            )
        if indf != self.num_beads:
            r_0 = self.r[indf, :]
            r_1 = self.r[indf - 1, :]
            test_r_1 = r_trial[indf - ind0 - 1, :]
            t3_0 = self.t3[indf, :]
            t3_1 = self.t3[indf - 1, :]
            test_t3_1 = t3_trial[indf - ind0 - 1, :]
            t2_0 = self.t2[indf, :]
            t2_1 = self.t2[indf - 1, :]
            test_t2_1 = t2_trial[indf - ind0 - 1, :]
            delta_energy_poly += self.bead_pair_dE_poly(
                r_0, r_1, test_r_1, t3_0, t3_1, test_t3_1, t2_0, t2_1,
                test_t2_1
            )
        return delta_energy_poly

    def bead_pair_dE_poly(
        self, r_0, r_1, test_r_1, t3_0, t3_1, test_t3_1, t2_0, t2_1, test_t2_1
    ):
        """Compute the change in polymer energy when moving a single bead pair.

        For the current and proposed states of the polymer, calculate change in
        position, as well as the parallel and perpendicular components of that
        change, of the bead pair.

        Calcualate the bend vectors for the existing and trial orientations.

        Calculate the change in energy given these properties using theory for
        a discretized worm-like chain.

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        r_1 : array_like (3,)
            Position vector of second bead in bend
        test_r_1 : array_like (3,)
            Position vector of second bead in bend (TRIAL MOVE)
        t3_0 : array_like (3,)
            t3 tangent vector of first bead in bend
        t3_1 : array_like (3,)
            t3 tangent vector of second bead in bend
        test_t3_1 : array_like (3,)
            t3 tangent vector of second bead in bend (TRIAL MOVE)
        t2_0 : array_like (3,)
            t2 tangent vector of first bead in bend
        t2_1 : array_like (3,)
            t2 tangent vector of second bead in bend
        test_t2_1 : array_like (3,)
            t2 tangent vector of second bead in bend (TRIAL MOVE)

        Returns
        -------
        delta_energy_poly : float
            Change in polymer energy associated with the move of a single
            bead pair.
        """
        delta_r_test = test_r_1 - r_0
        delta_r_par_test = np.dot(delta_r_test, t3_0)
        delta_r_perp_test = delta_r_test - delta_r_par_test * t3_0
        delta_r = r_1 - r_0
        delta_r_par = np.dot(delta_r, t3_0)
        delta_r_perp = delta_r - delta_r_par * t3_0
        bend_vec_test = test_t3_1 - t3_0 - self.eta * delta_r_perp_test
        bend_vec = t3_1 - t3_0 - self.eta * delta_r_perp
        delta_energy_poly = (
                0.5 * self.eps_bend / self.delta
                    * np.dot(bend_vec_test, bend_vec_test)
                + 0.5 * self.eps_par / self.delta
                    * (delta_r_par_test - self.delta * self.gamma) ** 2
                + 0.5 * self.eps_perp / self.delta
                    * np.dot(delta_r_perp_test, delta_r_perp_test)
                )
        delta_energy_poly -= (
                0.5 * self.eps_bend / self.delta
                    * np.dot(bend_vec, bend_vec)
                + 0.5 * self.eps_par / self.delta
                    * (delta_r_par - self.delta * self.gamma) ** 2
                + 0.5 * self.eps_perp / self.delta
                    * np.dot(delta_r_perp, delta_r_perp)
            )
        return(delta_energy_poly)

    def __str__(self):
        """Return string representation of the Chromatin Polymer.
        """
        return f"Polymer_Class<Chromatin>" + super(Chromatin, self).__str__()

    def _find_parameters(self, length_bead):
        """Look up elastic parameters of ssWLC for each bead_length.

        TODO: Verify and update documentation given `self.lp`.

        TODO: Check definition of `delta`.

        Note the following properties:

        - The persistence length of DNA (`lp_DNA_nm`) is 53 nm
        - Each base pair of DNA (`nm_per_bp`) is 0.34 nm along the double helix

        Begin by nondimensionalizing the length of each bead (`length_bead`),
        which is entered as a number of base pairs. The nondimensionalized
        length is stored in `length_ndim` and represents the number of
        persistence lengths between the beads.

        The spacing of each bead in nm (`delta`) is given by the number of base
        pairs between beads (`length_bead`) multiplied by the number of nm of
        each base pair (`np_per_bp`).

        Interpolate from the parameter table to get the physical parameters of
        the WLC model matching the discretization of the polymer.

        Then redimentionalize the physical parameters. When redimensionalizing,
        note that the number of persistence lengths per bead (`length_ndim`) is
        given by (bp per bead) * (nm per bp) / (persistence length DNA in nm).

        The number of simulation units per persistence length (`lp`) is given
        by (nm per persistence length) / (nm per bp).

        Parameters
        ----------
        length_bead : float
            Number of base pairs represented by each bead of the polymer

        Returns
        -------
        delta : float
            Bead spacing in units of simulation (in our case, this is bps)
        eps_bend : float
            Bending modulus
        eps_par : float
            Stretch modulus
        eps_perp: float
            Shear modulus
        gamma : float
            Ground-state segment length
        eta : float
            Bend-shear coupling
        """
        length_dim = length_bead / self.lp
        lp = self.lp
        delta = length_dim

        eps_bend = np.interp(length_dim, dss_params[:, 0], dss_params[:, 1]) \
            / length_dim
        gamma = np.interp(length_dim, dss_params[:, 0], dss_params[:, 2]) \
            * length_dim * lp
        eps_par = np.interp(length_dim, dss_params[:, 0], dss_params[:, 3]) \
            / (length_dim * lp ** 2)
        eps_perp = np.interp(length_dim, dss_params[:, 0], dss_params[:, 4]) \
            / (length_dim * lp ** 2)
        eta = np.interp(length_dim, dss_params[:, 0], dss_params[:, 5]) \
            / lp

        return delta, eps_bend, eps_par, eps_perp, gamma, eta

    @classmethod
    def straight_line_in_x(
        cls,
        name: str,
        num_beads: int,
        bead_length: float,
        **kwargs
    ):
        """Construct polymer initialized uniformly along the positve x-axis.

        Parameters
        ----------
        name : str
            Name of polymer being constructed
        num_beads : int
            Number of monomeric units of polymer
        bead_length : float or (N,) array_like of float
            The amount of polymer path length between this bead and the next
            bead. For now, a constant value is assumed (the first value if an
            array is passed).

        Returns
        -------
        Polymer
            Object representation of a polymer
        """
        r = np.zeros((num_beads, 3))
        r[:, 0] = bead_length * np.arange(num_beads)
        t3 = np.zeros((num_beads, 3))
        t3[:, 0] = 1
        t2 = np.zeros((num_beads, 3))
        t2[:, 1] = 1
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def arbitrary_path_in_x_y(
        cls,
        name: str,
        num_beads: int,
        bead_length: float,
        shape_func: Callable[[float], float],
        step_size: float = 0.001,
        **kwargs
    ):
        """Construct a polymer initialized as y = f(x) from x = 0.

        TODO: If we want to have variable linker lengths, the spacing of
        monomeric units must be different, and this must be accounted for
        when selecting x-positions of beads.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : float or (N,) array_like of float
            The amount of polymer path length between this bead and the next
            bead. For now, a constant value is assumed (the first value if an
            array is passed).
        shape_func : Callable[[float], float]
            Shape of the polymer where z = 0 and y = f(x)
        step_size : float
            Step size for numerical evaluation of contour length when
            domain of the shape function used to initialize polymer

        Returns
        -------
        Polymer
            Object representing a polymer following path y = f(x)
        """
        r = paths.coordinates_in_x_y(
            num_beads,
            bead_length,
            shape_func,
            step_size
        )
        t3, t2 = paths.get_tangent_vals_x_y(
            r[:, 0],
            shape_func,
            step_size
        )
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def arbitrary_path_in_x_y_z(
        cls,
        name: str,
        num_beads: int,
        bead_length: float,
        shape_func_x: Callable[[float], float],
        shape_func_y: Callable[[float], float],
        shape_func_z: Callable[[float], float],
        step_size: float = 0.001,
        **kwargs
    ):
        """Construct a polymer initialized as y = f(x) from x = 0.

        TODO: If we want to have variable linker lengths, the spacing of
        monomeric units must be different, and this must be accounted for
        when selecting x-positions of beads.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : float or (N,) array_like of float
            The amount of polymer path length between this bead and the next
            bead. For now, a constant value is assumed (the first value if an
            array is passed).
        shape_func_x, shape_func_y, shape_func_z : Callable[[float], float]
            Parametric functions to obtain the x, y, z coordinates of the path
        step_size : float
            Step size for numerical evaluation of contour length when
            domain of the shape function used to initialize polymer

        Returns
        -------
        Polymer
            Object representing a polymer following path y = f(x)
        """
        r, parameter_vals = paths.coordinates_in_x_y_z(
            num_beads,
            bead_length,
            shape_func_x,
            shape_func_y,
            shape_func_z,
            step_size
        )
        t3, t2 = paths.get_tangent_vals_x_y_z(
            parameter_vals,
            shape_func_x,
            shape_func_y,
            shape_func_z,
            step_size,
            r
        )
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def gaussian_walk_polymer(
        cls,
        name: str,
        num_beads: int,
        bead_length: float,
        **kwargs
    ):
        """Initialize a polymer to a Gaussian random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : float or (N,) array_like of float
            The amount of polymer path length between this bead and the next
            bead. For now, a constant value is assumed (the first value if an
            array is passed).

        Returns
        -------
        Polymer
            Object representing a polymer initialized as Gaussian random walk
        """
        r = paths.gaussian_walk(num_beads, bead_length)
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)


def sin_func(x: float) -> float:
    """Sine function to which the polymer will be initialized.

    Parameters
    ----------
    x : float
        Input to the shape function

    Returns
    -------
    float
        Output to the shape function
    """
    return 50 * np.sin(x / 35)


def helix_parametric_x(t: float) -> float:
    """Parametric equation for x-coordinates of a helix.

    Parameters
    ----------
    t : float
        Parameter input to the shape function

    Returns
    -------
    float
        Output to the shape function
    """
    x = 180 * np.cos(t)
    return x


def helix_parametric_y(t: float) -> float:
    """Parametric equation for y-coordinates of a helix.

    Parameters
    ----------
    t : float
        Parameter input to the shape function

    Returns
    -------
    float
        Output to the shape function
    """
    y = 180 * np.sin(t)
    return y


def helix_parametric_z(t: float) -> float:
    """Parametric equation for z-coordinates of a helix.

    Parameters
    ----------
    t : float
        Parameter input to the shape function

    Returns
    -------
    float
        Output to the shape function
    """
    z = 60 * t
    return z
