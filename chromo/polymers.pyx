# cython: profile=False

"""Polymers that will make up our simulation.
"""

import pyximport
pyximport.install()

# Built-in Modules
from typing import Callable
from libc.math cimport sin, cos

# External Modules
import numpy as np
cimport numpy as np
import pandas as pd
from scipy.special import comb

# Custom Modules
import chromo.beads as beads
import chromo.util.mc_stat as mc_stat
from chromo.util.linalg cimport (vec_sub3, vec_dot3, vec_scale3)
from .util import dss_params
from .util import poly_paths as paths
from .util.nucleo_geom import consts_dict

# Global type aliases and variables
ctypedef np.uint8_t uint8
cdef np.ndarray empty_2d = np.empty((0, 0))
cdef np.ndarray mty_2d_int = np.empty((0, 0), dtype=int)
cdef np.ndarray empty_1d = np.empty((0, ))
cdef np.ndarray empty_1d_str = np.empty((0, ), dtype=str)


cdef double E_HUGE = 1E25

# Length per base pair of DNA
cdef double LENGTH_BP = 0.332
# Natural twist of linker DNA (in radians / bp)
cdef double NATURAL_TWIST_BARE = 2 * np.pi / 10.5


cdef class TransformedObject:
    """Represents any object undergoing physical transformations during MC sim.

    Notes
    -----
    This class provides the transformation matrix attribute. The transformation
    matrix is updated with each iteration of the MC simulation. It's inclusion
    avoids costly matrix initialization with each iteration, drastically
    improving runtime.

    Attributes
    ----------
    transformation_mat : array_like (4, 4) of double
        Homogeneous transformation matrix for an arbitrary transformation in
        the MC simulation; stored as attribute to avoid costly array
        generation during inner loop of simulation
    """

    def __init__(self):
        self.transformation_mat = np.identity(4, dtype=np.double)


cdef class PolymerBase(TransformedObject):
    """Base class representation of an arbitrary polymer.

    Notes
    -----
    See `TransformedObject` documentation for additional notes and attributes.

    Attributes
    ----------
    name : str
        A name of the polymer for convenient representation; should be a
        valid filename
    log_path : str
        Path to configuration tracker log (if empty, flags for no configuration
        log to be generated)
    beads : beads.Bead
        Monomeric unit forming the polymer
    num_beads : long
        Number of monomeric units forming the polymer
    configuration_tracker : mc_stat.ConfigurationTracker
        Object tracking configuration of the polymer over the course of the
        MC simulation, used for purposes of tracking simulation convergence
    lp : double
        Persistence length of the polymer
    num_binders : long
        Number of reader protein types bound to the polymer which are tracked
        by the simulator
    n_binders_p1 : long
        One more than `num_binders` to avoid repeated calculation
    binder_names : array_like (M,) of str
        Name of M bound chemical states tracked in `states`; length of
        `binder_names` must match number of columns in `states`
    all_inds : array_like (N,) of long
        Indices corresponding to each bead in the polymer
    r, t3, t2 : array_like (N, 3) of double
        Current position (r), tangent vector (t3) and material normal (t2) in
        global coordinates (x, y, z) for each of N beads in the polymer
    r_trial, t3_trial, t2_trial : array_like (N, 3) of double
        Proposed position (r), tangent vector (t3) and material normal (t2) in
        global coordinates (x, y, z) for each of N beads in the polymer;
        evaluated for acceptance using energy change determined by `compute_dE`
    states : array_like (N, M) of long
        Current state of each of the M reader proteins being tracked for each
        of the N beads in the polymer
    states_trial : array_like (N, M) of long
        Trial state of each of the M reader proteins being tracked for each of
        the N beads in the polymer
    chemical_mods : array_like (N, M) of long
        States of M chemical modifications (e.g., counts of H3K9me3) on each
        of N beads in the polymer
    chemical_mod_names : array_like (M,) of str
        Name of M chemical modifications made to the polymer tracked for each
        bead, corresponding to columns in `chemical_mods`
    direction : array_like (3,) of double
        Vector describing the direction of a transformation during a move in
        the MC simulation; stored as attribute to avoid costly array generation
        during inner loop of simulation
    point : array_like (3,) of double
        Used to describe a reference point (typically a fulcrum) for a move
        in the MC simulation; stored as attribute to avoid costly array
        generation during inner loop of simulation
    last_amp_move : double
        Most recent move amplitude during simulation step
    last_amp_bead : long
        Most recent bead amplitude during simulation step
    required_attrs, _arrays, _3d_arrays : array_like (L,) of str
        Minimal required attributes (required_attrs), arrays tracked (_arrays),
        and multi-index arrays tracked (_3d_arrays) by the polymer
    dr, dr_perp, bend : array_like (3,) of double
        Change in position (dr), perpendicular component of the change in
        position (dr_perp), and bend vector (bend) used to compute the
        elastic energy of a bond in a polymer; stored as attribute to avoid
        costly array generation during inner loop of simulation
    dr_test, dr_perp_test, bend_test : array_like (3,) of double
        Change in position (dr_test), perpendicular component of the change in
        position (dr_perp_test), and bend vector (bend_test) of proposed
        configuration used to compute the elastic energy of a bond in a
        polymer; stored as attribute to avoid costly array generation during
        inner loop of simulation
    densities_temp : array_like (2, n_binders_p1, 8) of double
        Densities for current (dim0 = 0) and trial (dim0 = 1) bins containing
        bead during MC move; the densities are tracked for the bead itself
        and each bound reader proteins (stored in dim1); the densities are
        tracked for 8 bins containing the bead before and after the move
        (stored in dim2); stored as attribute to avoid costly array generation
        during inner loop of simulation
    max_binders : long
        Indicates the maximum number of total binders of any type that can bind
        a single bead. This attribute accounts for steric limitations to the
        total number of binders that can attach to any give bead in the polymer.
        A value of `-1` is reserved for no limit to the total number of binders
        bound to a bead. Any move which proposes more than the maximum number
        of binders is assigned an energy cost of 1E99 kT per binder beyond the
        limit.

    Notes
    -----
    Do not instantiate instances of `PolymerBase` -- this class functions as a
    cython equivalent of an abstract base class. `PolymerBase` specifies the
    attributes and methods required to define a new polymer class. The class
    also implements utility methods common to all polymer subclasses.
    """

    def __init__(
        self,
        str name,
        double[:, ::1] r = empty_2d,
        *,
        str log_path = "",
        double[:, ::1] t3 = empty_2d,
        double[:, ::1] t2 = empty_2d,
        double[:] bead_length = empty_1d,
        double lp = 0,
        long[:, ::1] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        long[:, ::1] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        long max_binders = -1
    ):
        """Any polymer class requires these key attributes.

        Notes
        -----
        Specifying these attributes ensures that new polymer classes are
        compatible with the rest of the codebase. Optional arguments must be
        specified by name.

        `required_attrs` lists the required attributes of any polymer class,
        `_arrays` lists the attributes of the polymer class which are stored as
        arrays, and `_3d_arrays` lists arrays with multi-indexed (x, y, z)
        values. I recognize that these would be better implemented as class
        attributes if we were working in pure Python, but for Cython, it is
        easier to treat these as instance attributes. This is because there is
        no `.__class__` attribute in Cython, and we need to be able to access
        `required_attrs`, `_arrays`, and `_3d_arrays` from the C code for any
        particular subclass using methods in the superclass.

        Parameters
        ----------
        name : str
            A name of the polymer for convenient representation; should be a
            valid filename
        r : array_like (N, 3) of double
            The positions of each bead forming monomeric units of the polymer
        t3 : Optional array_like (N, 3) of double
            The tangent vector to each bead in global coordinates (default is
            empty array)
        t2 : Optional array_like (N, 3) of double
            Material normal to each bead in the global coordinate system; this
            vector should be orthogonal to t2 (default is empty array)
        bead_length : array_like (N,) of double
            The length of each bond in the polymer (default is empty array)
        lp : Optional double
            The persistence length of the polymer (default is 0)
        states : Optional array_like (N, M) of int
            State of each of the M reader proteins being tracked for each of
            the N beads (default is empty array)
        binder_names : Optional array_like (M,) of str
            Name of M bound chemical states tracked in `states`; length of
            `binder_names` must match number of columns in `states` (default is
            empty array)
        chemical_mods : Optional array_like (N, M) of long
            States of M chemical modifications (e.g., counts of H3K9me3) on
            each of N beads in the polymer (default is empty array)
        chemical_mod_names : Optional array_like (M,) of str
            Name of M chemical modifications made to the polymer tracked for
            each bead, corresponding to columns in `chemical_mods` (default
            is empty array)
        log_path : Optional str
            Path to configuration tracker log (default is empty string,
            flagging for no configuration log to be generated)
        max_binders : Optional long
            Indicates the maximum number of total binders of any type that can
            bind a single bead. This attribute accounts for steric limitations
            to the total number of binders that can attach to any give bead in
            the polymer. A value of `-1` is reserved for no limit to the total
            number of binders bound to a bead. Any move which proposes more
            than the maximum number of binders is assigned an energy cost of
            1E99 kT per binder beyond the limit. (default = -1)
        """
        cdef long num_binders, num_beads
        cdef np.ndarray required_attrs, _arrays, _3d_arrays

        super(PolymerBase, self).__init__()

        self.name = name
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.bead_length = bead_length
        self.states = states
        self.max_binders = max_binders
        self.binder_names = binder_names
        self.chemical_mods = chemical_mods
        self.chemical_mod_names = chemical_mod_names
        self.num_beads = self.get_num_beads()
        self.fill_missing_arguments()
        self.all_inds = np.arange(0, self.num_beads, 1)
        self.num_binders = self.get_num_binders()
        self.n_binders_p1 = self.num_binders + 1
        self.log_path = log_path
        self.check_binders(states, binder_names)
        self.construct_beads()
        self.update_log_path(log_path)
        self.lp = lp
        self.required_attrs = np.array([
            "name", "r", "t3", "t2", "states", "binder_names", "num_binders",
            "beads", "num_beads", "lp", "bead_length"
        ])
        self._arrays = np.array(
            ['r', 't3', 't2', 'states', 'chemical_mods', 'bead_length']
        )
        self._3d_arrays = np.array(['r', 't3', 't2'])
        self._single_values = np.array(["name", "lp", "lt", "bp_wrap"])

        # For move proposals
        self.r_trial = self.r.copy()
        self.t3_trial = self.t3.copy()
        self.t2_trial = self.t2.copy()
        self.states_trial = self.states.copy()
        self.transformation_mat = np.empty((4, 4), dtype='d')
        self.last_amp_bead = 0
        self.last_amp_move = 0
        self.direction = np.zeros((3,), dtype='d')
        self.point = np.zeros((3,), dtype='d')

        # For elastic energy calculation
        self.dr = np.zeros((3,), dtype='d')
        self.dr_test = np.zeros((3,), dtype='d')
        self.dr_perp = np.zeros((3,), dtype='d')
        self.dr_perp_test = np.zeros((3,), dtype='d')
        self.bend = np.zeros((3,), dtype='d')
        self.bend_test = np.zeros((3,), dtype='d')

        # For field energy calculation
        self.densities_temp = np.zeros((2, self.n_binders_p1, 8), dtype='d')

    def fill_missing_arguments(self):
        """Fill empty arguments in `__init__` call.

        Fills t3, t2, states, binder_names, chemical_mods, chemical_mod_names
        if not defined during polymer initialization.
        """
        if np.size(self.t3) == 0:
            print("No t3 tangent vectors defined.")
            self.t3 = np.zeros((self.num_beads, 3), dtype='d')
        if np.size(self.t2) == 0:
            print("No t2 tangent vectors defined.")
            self.t2 = np.zeros((self.num_beads, 3), dtype='d')
        if np.size(self.states) == 0:
            print("No states defined.")
            self.states = np.zeros((self.num_beads, 1), dtype=int)
        if np.size(self.binder_names) == 0:
            self.binder_names = np.array(["null_reader"])
        if np.size(self.chemical_mods) == 0:
            print("No chemical modifications defined.")
            self.chemical_mods = np.zeros((self.num_beads, 1), dtype=int)
        if np.size(self.chemical_mod_names) == 0:
            self.chemical_mod_names = np.array(["null_mod"])

    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    ):
        """Compute the change in configurational (elastic) energy of a polymer.

        Notes
        -----
        This method must be implemented for any subclass of PolymerBase and
        is implemented based on the polymer model representing the subclass.

        Parameters
        ----------
        move_name : str
            Name of the MC move for which the energy change is being calculated
        inds : array_like of int
            Indices of the beads affected by the MC move
        n_inds : int
            Number of indices affected by the MC move

        Returns
        -------
        float
            Elastic energy change associated with the MC move
        """
        pass

    cdef void construct_beads(self):
        """Construct a list of beads forming the polymer.
        
        Notes
        -----
        This method must be implemented for any subclass of PolymerBase

        Stores a dictionary of beads constructing the polymer chain, where keys
        represent bead IDs and values represent `beads.Bead` objects. The
        dictionary is stored as an attribute of the `PolymerBase` class.
        """
        pass

    def __str__(self):
        """Return string representation of the Polymer.

        Returns
        -------
        str
            String representation listing key attributes of the polymer
        """
        rep = f"Polymer<{self.name}, nbeads={self.num_beads}, " \
            f"nbinder={self.num_binders}>"
        return rep

    def check_attrs(self):
        """Check that the required attributes are specified in the class.

        Notes
        -----
        The required attributes of any polymer are stored in `required_attrs`.
        All required attributes must be initialized for any polymer to be
        instantiated. This method is run during the initialization of the
        polymer to verify that all required attributes are specified.
        """
        cdef str attr
        cdef long i
        for i in range(len(self.required_attrs)):
            attr = str(self.required_attrs[i])
            if not hasattr(self, attr):
                raise NotImplementedError(
                    "Polymer subclass missing required attribute: " + attr
                )
        if self.lp == 0:
            raise ValueError(
                "Specify the persistence length in the subclass of Polymer"
            )

    cdef void check_binders(self, long[:, ::1] states, np.ndarray binder_names):
        """Verify that specified binder states and names are valid.

        Parameters
        ----------
        states : array_like (N, M) of long
            State of each of the M binders being tracked for each of the N bead
        binder_names : array_like (M,) of str
            Name of M bound chemical states tracked in `states`; the number of
            `binder_names` specified must match number of columns in `states`
        """
        cdef long n_beads, num_binders
        if states.shape[1] is not 0:
            num_beads = states.shape[0]
            num_binders = states.shape[1]
            if num_binders != len(binder_names):
                raise ValueError("Each chemical state must be given a name.")
            if num_beads != len(self.r):
                raise ValueError("Initial epigenetic state of wrong length.")

    cdef void check_chemical_mods(
        self,
        long[:, ::1] chemical_mods,
        np.ndarray chemical_mod_names
    ):
        """Check that a valid array of chemical modifications was specified.
        
        Notes
        -----
        There must be a sequence of chemical modifications per reader protein
        included in the model. The sequence of chemical modifications must be
        of length `self.num_beads`, meaning there is a chemical modification
        status specified for each bead of the polymer.

        Parameters
        ----------
        chemical_mods : array_like (N, M) of long
            States of M chemical modifications (e.g., counts of H3K9me3) on
            each of N beads in the polymer
        chemical_mod_names : Optional (M, ) str or Sequence[str]
            Name of chemical modifications made to the polymer tracked for each
            bead in `chemical_mods` attribute
        """
        if self.num_binders != chemical_mods.shape[1]:
            raise ValueError(
                "Number of chemical modifications and binders must match."
            )
        if self.num_beads != chemical_mods.shape[0]:
            raise ValueError(
                "Chemical mod. state must be specified for each polymer bead."
            )
        if len(chemical_mod_names) != chemical_mods.shape[1]:
            raise ValueError(
                "Please specify a name for each chemical modification."
            )
    
    @staticmethod
    def load_seqs(paths_to_seqs: np.ndarray) -> np.ndarray:
        """Load sequence of all chemical modifications on beads of polymer.

        Parameters
        ----------
        paths_to_seqs : array_like (M,) of str
            Paths to the sequence of chemical modifications or binders on the
            beads. A single path should be provided per epigenetic mark.

        Returns
        -------
        array_like (N, M) of long
            States of M chemical modifications (e.g., counts of H3K9me3) on
            each of N beads in the polymer
        """
        num_mods = len(paths_to_seqs)
        if num_mods == 0:
            return mty_2d_int
        first_seq = np.loadtxt(paths_to_seqs[0], dtype=float)
        first_seq = np.round(first_seq).astype(int)
        num_beads = first_seq.shape[0]
        chemical_mods = np.zeros((num_beads, num_mods), dtype=int)
        chemical_mods[:, 0] = first_seq
        for i in range(1, num_mods):
            path = paths_to_seqs[i]
            chemical_mod = np.loadtxt(path, dtype=float)
            chemical_mod = np.round(chemical_mod).astype(int)
            if chemical_mod.shape[0] != num_beads:
                raise ValueError(
                    "Inconsistent sequences of chemical modifications"
                )
            chemical_mods[:, i] = chemical_mod
        return chemical_mods
    
    cpdef void update_log_path(self, str log_path):
        """Update the path to the configuration tracker log.
        
        Notes
        -----
        The configuration tracker is used during the simulation to evaluate 
        structural convergence of the polymer.

        Parameters
        ----------
        log_path : str
            Path to the configuration tracker log. If None, no configuration
            tracker will be initialized. (default = None)
        """
        if log_path != "":
            self.configuration_tracker = \
                mc_stat.ConfigurationTracker(log_path, self.r)

    cpdef np.ndarray get_prop(self, long[:] inds, str prop):
        """Get specified property of beads at listed indices.

        Parameters
        ----------
        inds : array_like (L,) of long
            Bead indices at which to isolate multi-indexed attributes (e.g., 
            position, orientation) of the polymer
        prop : str
            Property of the bead to return at specified indices

        Returns
        -------
        np.ndarray (M, 3)
            Specified property of beads at specified indices
        """
        return np.array(
            [self.beads[i].__dict__[prop] for i in inds]
        )

    cpdef np.ndarray get_all(self, str prop):
        """Get some bead property value from all beads.
        
        Notes
        -----
        Generates an array from a specified attribute separately stored in 
        each bead object representing the polymer. This utility function is 
        useful for accessing attributes which are separately stored in bead 
        objects and not aggregated in the polymer object. 

        Parameters
        ----------
        prop : str
            Name of the property to obtain from all beads

        Returns
        -------
        np.ndarray
            Data frame of property values from all beads (default = None)
        """
        array = np.array(
            [self.beads[i].__dict__[prop] for i in range(len(self.beads))]
        )
        if len(array.shape) == 1:
            array = np.atleast_2d(array).T
        return array

    def update_prop(self, str prop):
        """Update bead property for each bead in polymer.

        Notes
        -----
        This utility function copies an attribute of the polymer to each bead
        forming the polymer.

        Parameters
        ----------
        prop : str
            Name of the bead property being updated
        """
        cdef long i
        for i in range(len(self.beads)):
            self.beads[i].__dict__[prop] = self.getattr(self, prop)[i]

    def to_dataframe(self) -> pd.DataFrame:
        """Write canonical CSV representation of the Polymer to file.

        Notes
        -----
        The top-level multi-index values for the columns should correspond to
        the kwargs, to simplify unpacking the structure (and to make for easy
        accessing of the dataframe).

        First get a listing of all the parts of the polymer that have to be
        saved. These are defined in the class's `_arrays` attribute.

        Next, separate out the arrays into two types: "vector" arrays are
        composed of multiple columns (like r) and so we need to build a multi-
        index for putting them into the data frame correctly.

        All other arrays can be added as a single column to the data frame.

        ReaderProtein names is a special case because it does not fit with
        dimensions of the other properties, which are saved per bead.

        Construct the parts of the DataFrame that need a multi-index.

        TODO: remove "list" after numpy fixes issue 17305:
        https://github.com/numpy/numpy/issues/17305

        Replace:
            `vector_arr = np.concatenate(list(vector_arrs.values()), axis=1)`
        with:
            `vector_arr = np.concatenate(vector_arrs.values(), axis=1)`

        After adding multi-index properties, add remaining arrays one-by-one.

        Returns
        -------
        pd.DataFrame
            Data frame representation of the polymer, as recorded in the CSV
            file that was generated
        """
        arrays = {
            name: getattr(self, name) for name in self._arrays
            if hasattr(self, name)
        }
        single_vals = {
            name: getattr(self, name) for name in self._single_values
            if hasattr(self, name)
        }
        vector_arrs = {}
        regular_arrs = {}

        for name, arr in arrays.items():
            if name in self._3d_arrays:
                vector_arrs[name] = arr
            elif name != 'states' and name != 'chemical_mods' \
                    and name != "bead_length":
                regular_arrs[name] = arr
        vector_arr = np.concatenate(list(vector_arrs.values()), axis=1)
        vector_index = pd.MultiIndex.from_product(
            [vector_arrs.keys(), ('x', 'y', 'z')]
        )
        vector_df = pd.DataFrame(vector_arr, columns=vector_index)
        if len(self.chemical_mod_names) > 0:
            states_index = pd.MultiIndex.from_tuples(
                [('states', name) for name in self.binder_names]
            )
            chem_mod_index = pd.MultiIndex.from_tuples(
                [('chemical_mods', name) for name in self.chemical_mod_names]
            )
            states_df = pd.DataFrame(
                np.asarray(self.states), columns=states_index
            )
            chemical_mods_df = pd.DataFrame(
                np.asarray(self.chemical_mods), columns=chem_mod_index,
                dtype=int
            )
            df = pd.concat([vector_df, states_df, chemical_mods_df], axis=1)
        else:
            df = vector_df
        # Store bead lengths
        df["bead_length"] = np.broadcast_to(
            np.append(self.bead_length, 0).reshape((-1, 1)), (len(df.index), 1)
        )
        # Store scalar measure of the maximum number of beads
        for name, arr in regular_arrs.items():
            if name == "max_binders":
                df[name] = np.broadcast_to(arr, (len(df.index), 1))
            else:
                df[name] = arr
        for name, val in single_vals.items():
            arr_temp = np.empty((len(df.index), 1), dtype=object)
            arr_temp[0, 0] = str(val)
            df[name] = arr_temp
            df[name] = df[name].apply(lambda x: x if x is not None else "")
        return df

    def to_csv(self, str path) -> pd.DataFrame:
        """Save Polymer object to CSV file as DataFrame.

        Parameters
        ----------
        path : str
            Path to file at which to save CSV representation of the polymer

        Returns
        -------
        pd.DataFrame
            Data frame representation of the polymer, as recorded in the CSV
            file that was generated
        """
        return self.to_dataframe().to_csv(path)

    def to_file(self, str path) -> pd.DataFrame:
        """Synonym for `to_csv` to conform to `make_reproducible` spec.

        Notes
        -----
        See documentation for `to_csv` for details.
        """
        return self.to_csv(path)

    @classmethod
    def from_csv(cls, str csv_file) -> pd.DataFrame:
        """Construct Polymer from CSV file.

        Parameters
        ----------
        csv_file : str
            Path to CSV file from which to construct polymer

        Returns
        -------
        PolymerBase
            Object representation of a polymer
        """
        df = pd.read_csv(csv_file)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df, name=None, **kwargs):
        """Construct Polymer object from DataFrame; inverts `.to_dataframe`.

        Notes
        -----
        TODO: For the DetailedChromatinWithSterics class, this method does not
        TODO: load binders from file. Binders must be specified separately as
        TODO: kwargs. Perhaps a binders file can be handled in the future.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame representation of the polymer
        name : Optional str
            Name of the polymer to be formed (default is None)

        Returns
        -------
        PolymerBase
            Polymer object reflecting the dataframe
        """
        # top-level multiindex values correspond to kwargs
        kwnames = np.unique(df.columns.get_level_values(0))
        kwargs_ = {
            name: np.ascontiguousarray(df[name].to_numpy()) for name in kwnames
        }
        # extract names of each epigenetic state from multi-index
        if 'states' in df:
            binder_names = df['states'].columns.to_numpy()
            kwargs_['binder_names'] = binder_names
        if 'chemical_mods' in df:
            chemical_mod_names = df['chemical_mods'].columns.to_numpy()
            kwargs_['chemical_mod_names'] = chemical_mod_names
        if 'max_binders' in df:
            kwargs_['max_binders'] = kwargs_['max_binders'][0]
        if "name" in df and name is None:
            kwargs_['name'] = df['name'][0]
        elif name is None:
            kwargs_['name'] = "unnamed"
        else:
            kwargs_['name'] = name
        if "lp" in df:
            kwargs_['lp'] = float(kwargs_['lp'][0])
        if "lt" in df:
            kwargs_['lt'] = float(kwargs_['lt'][0])
        if "bp_wrap" in df:
            kwargs_['bp_wrap'] = float(kwargs_['bp_wrap'][0])
        if "bead_length" in df:
            kwargs_['bead_length'] = \
                np.asarray(kwargs_['bead_length'][:-1]).flatten().astype(float)
        for key, value in kwargs.items():
            kwargs_[key] = value
        return cls(**kwargs_)

    @classmethod
    def from_file(cls, path, name=None, **kwargs):
        """Construct Polymer object from string representation.

        Notes
        -----
        TODO: For the DetailedChromatinWithSterics class, this method does not
        TODO: load binders from file. Binders must be specified separately as
        TODO: kwargs. Perhaps a binders file can be handled in the future.

        Parameters
        ----------
        path : str
            Path to CSV file representing the polymer
        name : Optional str
            Name of the polymer to be formed (default is None)

        Returns
        -------
        PolymerBase
            Polymer object reflecting the dataframe
        """
        if name is None:
            name = path.split("/")[-1].split(".")[0]
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        return cls.from_dataframe(df, name, **kwargs)

    cpdef long get_num_binders(self):
        """Return number of states tracked per bead.
        
        Returns
        -------
        long
            Number of binder types bound to the polymer
        """
        return self.states.shape[1]

    cpdef long get_num_beads(self):
        """Return number of beads in the polymer.
        
        Returns
        -------
        long
            Number of beads forming the polymer
        """
        return self.r.shape[0]

    cpdef bint is_field_active(self):
        """Evaluate if the polymer is affected by a field.
        
        Notes
        -----
        In the future, you can make the code more efficient by activating the
        field only when necessary. The field is necessary only when the polymer
        is bound by reader proteins OR is in a confinement OR has a volume
        fraction limit OR has a non-specific chi interaction. For now, to be
        safe, we will assume that the field always needs to be assessed.

        Returns
        -------
        bint
            Flag indicating whether the polymer is affected by the field (True)
            or is agnostic to the field (False)
        """
        return 1


cdef class Rouse(PolymerBase):
    """Class representation of Rouse polymer.

    Notes
    -----
    The Rouse model represents flexible chains of beads connected by harmonic
    springs. The model inherits from the PolymerBase class, which provides the
    basic attributes and functionality shared by all polymers. Please see
    `PolymerBase` documentation for descriptions of attributes.
    """

    def __init__(
        self,
        str name,
        double[:, ::1] r,
        *,
        double lp,
        np.ndarray[long, ndim=2] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        np.ndarray[long, ndim=2] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = ""
    ):
        """Initialize the Rouse polymer.

        Notes
        -----
        Please refer to documentation for `PolymerBase.__init__()` method for
        description of shared parameters.

        The Rouse model is not yet implemented in this codebase. When the
        model is implemented, be sure to remove the `NotImplementedError`
        statement.

        Parameters
        ----------
        lp : double
            Persistence length of the polymer
        """
        super(Rouse, self).__init__(
            name, r, states=states, binder_names=binder_names,
            log_path=log_path, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names
        )
        self.lp = lp
        raise NotImplementedError(
            "The Rouse polymer model is not yet implemented"
        )

    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    ):
        """Compute the change in configurational energy of a polymer.
        
        Notes
        -----
        This method currently serves as a place-holder. The Rouse polymer is 
        not yet implemented into this codebase.
        
        Please see documentation for `PolymerBase.compute_dE()` for 
        description of parameters and return arguments.
        """
        pass

    cdef void construct_beads(self):
        """Construct a list of beads forming the polymer.
        
        Notes
        -----
        This method currently serves as a place-holder. The Rouse polymer is 
        not yet implemented into this codebase.
        """
        pass


cdef class SSWLC(PolymerBase):
    """Class representation of a stretchable, shearable wormlike chain.

    Notes
    -----
    The `SSWLC` class describes a discrete, semiflexible polymer with
    attributes for position, orientation, chemical modification, and reader
    protein binding state.

    The polymer carries around a set of coordinates `r` of shape
    `(num_beads, 3)`, sets of material normals `t3` and `t2`, some number of
    chemical modifications, and some number of chemical binding states per
    bead. A third set of material normals `t1` can be derived from the
    cross-product of the `t2` and `t3`.

    Polymer also carries around a dictionary of beads of length `num_beads`.
    Each bead has a set of coordiantes `r` of length (3,). The beads also
    have material normals `t3` and `t2`. The beads optionally carry around
    some number of chemical modifications and binding states.

    If this codebase is used to simulate DNA, information about the chemical
    properties of each reader protein are stored in `ReaderProtein` objects. To
    model arbitrary SSWLC's, the different types of `Binder` objects may be
    specified to characterize bound components.

    See documentation for `PolymerBase` class for additional paramete details.

    TODO: allow differential discretization, and decay to constant
    discretization naturally.

    Parameters
    ----------
    delta : double
        Spacing between beads of the polymer, non-dimensionalized by
        persistence length
    eps_bend : double
        Bending modulus of polymer (obtained from `_find_parameters`)
    eps_par : double
        Stretch modulus of polymer (obtained from `_find_parameters`)
    eps_perp : double
        Shear modulus of polymer (obtained from `_find_parameters`)
    gamma : double
        Ground-state segment compression (obtained from `_find_parameters`)
    eta : double
        Bend-shear coupling (obtained from `_find_parameters`)
    bead_length : np.ndarray (N-1,) of float
        Dimensional spacing between beads of the discrete polymer (in nm)
    bead_rad : double
        Dimensional radius of each bead in the polymer (in nm)
    """

    def __init__(
        self, str name, double[:,::1] r, *, double[:] bead_length, double lp,
        double bead_rad=5, double[:,::1] t3=empty_2d,
        double[:,::1] t2=empty_2d, long[:,::1] states=mty_2d_int,
        np.ndarray binder_names=empty_1d, long[:,::1] chemical_mods=mty_2d_int,
        np.ndarray chemical_mod_names=empty_1d, str log_path = "",
        long max_binders = -1
    ):
        """Construct a `SSWLC` polymer object as a subclass of `PolymerBase`.

        Notes
        -----
        Please refer to documentation for `PolymerBase.__init__()` method for
        description of shared parameters.

        Parameters
        ----------
        bead_length : double[:]
            The amount of polymer path length between subsequent beads (in nm),
            defined for each linker along the polymer
        lp : double
            Dimensional persistence length of the SSWLC (in nm)
        bead_rad : double
            Radius of individual beads on the polymer (in nm)
        """
        cdef np.ndarray _arrays
        super(SSWLC, self).__init__(
            name, r, t3=t3, t2=t2, states=states, binder_names=binder_names,
            bead_length=bead_length, lp=lp, log_path=log_path,
            chemical_mods=chemical_mods, chemical_mod_names=chemical_mod_names,
            max_binders=max_binders
        )
        self.bead_rad = bead_rad
        self.construct_beads()
        self._find_parameters(self.bead_length)
        self.required_attrs = np.array([
            "name", "r", "t3", "t2", "states", "binder_names", "num_binders",
            "beads", "num_beads", "lp", "bead_rad"
        ])
        self._arrays = np.array([
            'r', 't3', 't2', 'states', 'bead_length', 'chemical_mods',
            'max_binders'
        ])
        self.check_attrs()
        self.mu_adjust_factor = 1

    cdef void construct_beads(self):
        """Construct `GhostBead` objects forming beads of the polymer.
        """
        self.beads = {
            i: beads.GhostBead(
                id_=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                states=self.states[i],
                binder_names=self.binder_names,
                rad=self.bead_rad
            ) for i in range(len(self.r))
        }

    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    ):
        """Compute change in elastic energy for proposed transformation.
        
        Notes
        -----
        Non-zero elastic energy changes occur at bonds of the polymer 
        directly changed by an MC move. If the indices affected by the MC 
        move are continuous, then the elastic energy change is computed for 
        the bonds at the boundaries of the continuous interval; internal 
        bonds are unchanged by such moves. When the indices affected by an
        MC move are non-continuous (i.e., the move is not applied to a 
        continuous polymer segment), then the elastic energy change is 
        calculated individually at each affected bead.

        Parameters
        ----------
        move_name : str
            Name of the MC move for which the energy change is being calculated
        inds : array_like (N, 3)
            Indices of N beads affected by the MC move
        n_inds : long
            Number of beads affected by the MC move

        Returns
        -------
        delta_energy_poly : double
            Change in polymer elastic energy assocaited with the trial move
        """
        cdef double delta_energy_poly
        cdef long ind0, indf, ind, i

        delta_energy_poly = 0

        if move_name == "change_binding_state":
            ind0 = inds[0]
            indf = inds[n_inds-1] + 1
            delta_energy_poly += self.binding_dE(ind0, indf, n_inds)
        
        elif (
            move_name == "slide" or move_name == "end_pivot" or
            move_name == "crank_shaft"
        ):
            ind0 = inds[0]
            indf = inds[n_inds-1] + 1
            delta_energy_poly += self.continuous_dE_poly(ind0, indf)
        
        elif move_name == "tangent_rotation":
            for i in range(n_inds):
                ind = inds[i]
                ind0 = ind
                indf = ind + 1
                delta_energy_poly += self.continuous_dE_poly(ind0, indf)

        return delta_energy_poly

    cdef double continuous_dE_poly(
        self,
        long ind0,
        long indf,
    ):
        """Compute change in elastic energy for a continuous bead region.
        
        Notes
        -----
        The internal configuration of a continuous segment selected for a move
        is unaffected; therefore, change in polymer energy can be determined
        from the beads at the ends of the selected segment.

        If a bound of the selected segment exists at the end of the polymer,
        then that bound does not contribute to a change in elastic energy, 
        since there are no bonds continuing beyond an end bead.

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

        Returns
        -------
        delta_energy_poly : double
            Change in energy of polymer associated with trial move
        """
        cdef long ind0_m_1, indf_m_1
        cdef double delta_energy_poly

        ind0_m_1 = ind0 - 1
        indf_m_1 = indf - 1

        delta_energy_poly = 0
        if ind0 != 0:
            delta_energy_poly += self.bead_pair_dE_poly_forward(
                self.r[ind0_m_1, :],
                self.r[ind0, :],
                self.r_trial[ind0, :],
                self.t3[ind0_m_1, :],
                self.t3[ind0, :],
                self.t3_trial[ind0, :],
                bond_ind = ind0_m_1
            )
        if indf != self.num_beads:
            delta_energy_poly += self.bead_pair_dE_poly_reverse(
                self.r[indf_m_1, :],
                self.r_trial[indf_m_1, :],
                self.r[indf, :],
                self.t3[indf_m_1, :],
                self.t3_trial[indf_m_1, :],
                self.t3[indf, :],
                bond_ind = indf_m_1
            )

        return delta_energy_poly

    cdef double E_pair(
        self, double[:] bend, double dr_par, double[:] dr_perp, long bond_ind
    ):
        """Calculate elastic energy for a pair of beads.

        Parameters
        ----------
        bend : double[:]
            Bending vector
        dr_par : double
            Magnitude of the parallel component of the displacement vector
        dr_perp : double[:]
            Perpendicular component of the displacement vector
        bond_ind : long
            Index of the bond between the beads

        Returns
        -------
        double
            Elastic energy of bond between the bead pair
        """
        cdef double E
        E = (
            0.5 * self.eps_bend[bond_ind] * vec_dot3(bend, bend) +
            0.5 * self.eps_par[bond_ind] * (dr_par - self.gamma[bond_ind])**2 +
            0.5 * self.eps_perp[bond_ind] * vec_dot3(dr_perp, dr_perp)
        )
        return E

    cdef double bead_pair_dE_poly_forward(
        self,
        double[:] r_0,
        double[:] r_1,
        double[:] test_r_1,
        double[:] t3_0,
        double[:] t3_1,
        double[:] test_t3_1,
        long bond_ind
    ):
        """Compute change in polymer energy when affecting a single bead pair.
        
        Notes
        -----
        For the current and proposed states of the polymer, calculate change in
        position, as well as the parallel magnitude and perpendicular component
        of that change, of the bead pair. Calcualate the bend vectors for the
        existing and trial orientations. Calculate the change in energy given
        these properties using theory for a discretized worm-like chain.
        
        In this 'forward' setup, the bead at index zero is linearly adjacent to
        the affected beads.

        This function can be written in a cleaner (slower) manner, as copied
        below. We have opted for in-place operations where possible to improve
        runtime. This is particularly important because this method is called
        from the inner-most loop of the MC algorithm.

        # Cleaner (and slower) code:
        cdef double de_poly, dr_par, dr_par_test
        cdef long i
        cdef double[:] dr, dr_test, dr_perp, dr_perp_test, bend, bend_test
        dr_test = vec_sub3(test_r_1, r_0)
        dr = vec_sub3(r_1, r_0)
        dr_par_test = vec_dot3(t3_0, dr_test)
        dr_par = vec_dot3(t3_0, dr)
        dr_perp_test = vec_sub3(dr_test, vec_scale3(t3_0, dr_par_test))
        dr_perp = vec_sub3(dr, vec_scale3(t3_0, dr_par))
        bend_test = vec_add3(
            test_t3_1, vec_sub3(
                vec_scale3(t3_0, -1.0), vec_scale3(
                    dr_perp_test, self.eta
                )
            )
        )
        bend = vec_add3(
            t3_1, vec_sub3(
                vec_scale3(t3_0, -1.0), vec_scale3(
                    dr_perp, self.eta
                )
            )
        )
        return self.E_pair(bend_test, dr_par_test, dr_perp_test) -\
            self.E_pair(bend, dr_par, dr_perp)

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
        bond_ind : int
            Index of the bond being affected

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef long i
        cdef double dr_par, dr_par_test
        
        for i in range(3):
            self.dr_test[i] = test_r_1[i] - r_0[i]
            self.dr[i] = r_1[i] - r_0[i]
        dr_par_test = vec_dot3(t3_0, self.dr_test)
        dr_par = vec_dot3(t3_0, self.dr)

        for i in range(3):
            self.dr_perp_test[i] = self.dr_test[i] - t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                test_t3_1[i] - t3_0[i] - self.dr_perp_test[i] *
                self.eta[bond_ind]
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[bond_ind]
            )
        return self.E_pair(
            self.bend_test, dr_par_test, self.dr_perp_test, bond_ind
        ) - self.E_pair(
            self.bend, dr_par, self.dr_perp, bond_ind
        )

    cdef double bead_pair_dE_poly_reverse(
        self,
        double[:] r_0,
        double[:] test_r_0,
        double[:] r_1,
        double[:] t3_0,
        double[:] test_t3_0,
        double[:] t3_1,
        long bond_ind
    ):
        """Compute change in polymer energy when affecting a single bead pair.
        
        Notes
        -----
        For the current and proposed states of the polymer, calculate change in
        position, as well as the parallel magnitude and perpendicular component
        of that change, of the bead pair. Calcualate the bend vectors for the
        existing and trial orientations. Calculate the change in energy given
        these properties using theory for a discretized worm-like chain.

        In this 'reverse' setup, the bead at index one is linearly adjacent to
        the affected beads.

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        test_r_0 : array_like (3,)
            Position vector of first bead in bend (TRIAL MOVE)
        r_1 : array_like (3,)
            Position vector of second bead in bend
        t3_0 : array_like (3,)
            t3 tangent vector of first bead in bend
        test_t3_0 : array_like (3,)
            t3 tangent vector of first bead in bend (TRIAL MOVE)
        t3_1 : array_like (3,)
            t3 tangent vector of second bead in bend
        bond_ind : int
            Index of the bond being affected

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef double dr_par, dr_par_test
        cdef long i

        for i in range(3):
            self.dr_test[i] = r_1[i]  - test_r_0[i]
            self.dr[i] = r_1[i] - r_0[i]
        dr_par_test = vec_dot3(test_t3_0, self.dr_test)
        dr_par = vec_dot3(t3_0, self.dr)
        for i in range(3):
            self.dr_perp_test[i] = self.dr_test[i] - test_t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                t3_1[i] - test_t3_0[i] - self.dr_perp_test[i] *
                self.eta[bond_ind]
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[bond_ind]
            )
        return self.E_pair(
            self.bend_test, dr_par_test, self.dr_perp_test, bond_ind
        ) - self.E_pair(
            self.bend, dr_par, self.dr_perp, bond_ind
        )

    cpdef double compute_E(self):
        """Compute the overall polymer energy at the current configuration.
        
        Notes
        -----
        This method loops over each bond in the polymer and calculates the 
        energy change of that bond.

        Returns
        -------
        double
            Configurational energy of the polymer.
        """
        cdef double E, dr_par
        cdef long i, j, i_m1
        cdef double[:] dr, dr_perp, bend
        cdef double[:] r_0, r_1, t3_0, t3_1

        E = 0
        for i in range(1, self.num_beads):
            i_m1 = i - 1
            r_0 = self.r[i_m1, :]
            r_1 = self.r[i, :]
            t3_0 = self.t3[i_m1, :]
            t3_1 = self.t3[i, :]

            dr = vec_sub3(r_1, r_0)
            dr_par = vec_dot3(t3_0, dr)
            dr_perp = vec_sub3(dr, vec_scale3(t3_0, dr_par))
            bend = t3_1.copy()
            for j in range(3):
                bend[j] += -t3_0[j] - self.eta[i_m1] * dr_perp[j]
            E += self.E_pair(bend, dr_par, dr_perp, i_m1)
        return E

    cdef double binding_dE(self, long ind0, long indf, long n_inds):
        """Compute energy difference associated with change in binding state.

        Parameters
        ----------
        ind0, indf : long
            The first (ind0) and last (indf) indices of the continuous 
            interval for which the binding state was changed
        n_inds : long
            Number of beads affected by binding state move

        Returns
        -------
        double
            Change in energy associated with the proposed binding state move
        """
        cdef long i, ind
        cdef double dE
        
        dE = 0
        for i in range(n_inds):
            ind = ind0 + i
            dE += self.bead_binding_dE(ind, self.states_trial[ind, :])
        return dE

    cdef double bead_binding_dE(self, long ind, long[:] states_trial_ind):
        """Compute energy difference for a single bead change in binding state.
        
        Notes
        -----
        Begin by identifying the chemical modification and binding states of the
        nucleosome. Using a single-site partition function, we compute the
        expected binding energy over all possible binding configurations.
        Previously, we had assumed that binders will first bind marked sites
        before binding unmarked sites. Using the single-site partition function,
        we remove this assumption and consider the entropy of binding.
        
        Consider that we are arranging `Nb` binders on `Nn` sites, of which `Nm`
        are marked. We can have up to `min(Nb, Nm)` binders on marked sites. For
        each value `i` in range(`min(Nb, Nm)`), there are `comb(Nm, i)` ways to
        arrange `i` binders on marked sites and `comb(Nn-Nm, Nb-i)` ways to
        arrange the remaining `Nb-i` binders on unmarked sites. Thus, the total
        number of ways to arrange `Nb` binders on `Nn` sites, of which `Nm` are
        marked, is the sum of `comb(Nm, i) * comb(Nn-Nm, Nb-i)` for `i` in
        range(`min(Nb, Nm)`). This is equivalent to `comb(Nn, Nb)`.
        
         The energy of each configuration is determined by the number of binders
        on marked sites, or `i`. If there is an energy `E_b` associated with
        each binder on a marked site, then the total energy of the configuration
        is `i * E_b`. Therefore, the site partition function representing
        binding is given by `sum(comb(Nm, i) * comb(Nn-Nm, Nb-i) * exp(-i * E_b))`
        for `i` in range(`min(Nb, Nm)`).
        
        `self.max_binders` indicates the maximum number of total binders of any
        type that can bind a single bead. This attribute accounts for steric
        limitations to the total number of binders that can attach to any give
        bead in the polymer. A value of `-1` is reserved for no limit to the
        total number of binders bound to a bead. Any move which proposes more
        than the maximum number of binders is assigned an energy cost of 1E99
        kT per binder beyond the limit.

        Parameters
        ----------
        ind : long
            Bead index for which the change in binding state is proposed
        states_trial_ind : array_like (M, )
            Proposed binding states for the bead affected by the move
        
        Returns
        -------
        double
            The energy change from the single bead's change in binding state
        """
        cdef long h, Nn, Nm, Nu
        cdef long tot_bound, diff
        cdef double dE
        cdef long[:] mod_states, states_ind

        mod_states = self.chemical_mods[ind]
        states_ind = self.states[ind].copy()
        dE = 0

        # Evaluate constraint on the total number of binders bound to a bead
        if self.max_binders != -1:

            # Trail configuration
            tot_bound = 0
            for h in range(self.num_binders):
                tot_bound += states_trial_ind[h]
            if tot_bound > self.max_binders:
                diff = tot_bound - self.max_binders
                dE += E_HUGE * diff

            # Current configuration
            tot_bound = 0
            for h in range(self.num_binders):
                tot_bound += states_ind[h]
            if tot_bound > self.max_binders:
                diff = tot_bound - self.max_binders
                dE -= E_HUGE * diff

        # Evaluate the change in binding energy
        for b in range(self.num_binders):
            # Number of sites
            Nn = self.beads[ind].binders[b].sites_per_bead
            # Number of modified tails
            Nm = mod_states[b]
            # Number of unmodified tails
            Nu = Nn - Nm
            # Single-site Helmholtz free energy (trial configuration)
            i = np.arange(states_trial_ind[b] + 1)
            dE += -np.log(np.sum(
                comb(Nm, i) *
                comb(Nn - Nm, states_trial_ind[b] - i) *
                (np.exp(
                    - (
                        i * self.beads[ind].binders[b].bind_energy_mod +
                        (states_trial_ind[b] - i) *
                        self.beads[ind].binders[b].bind_energy_no_mod
                    )
                ))
            ))
            # Single-site Helmholtz free energy (current configuration)
            i = np.arange(states_ind[b] + 1)
            dE -= -np.log(np.sum(
                comb(Nm, i) *
                comb(Nn - Nm, states_ind[b] - i) *
                (np.exp(
                    - (
                        i * self.beads[ind].binders[b].bind_energy_mod +
                        (states_ind[b] - i) *
                        self.beads[ind].binders[b].bind_energy_no_mod
                    )
                ))
            ))
            # Chemical potential
            # Note: Simulated annealing affects positive and negative chemical
            # potentials in different ways
            # TODO: Accommodate different types of simulated annealing schedules
            if self.beads[ind].binders[b].chemical_potential > 0:
                dE -= states_trial_ind[b] * (
                    self.beads[ind].binders[b].chemical_potential *
                    (-self.mu_adjust_factor + 2)
                )
                dE += states_ind[b] * (
                    self.beads[ind].binders[b].chemical_potential *
                    (-self.mu_adjust_factor + 2)
                )
            else:
                dE -= states_trial_ind[b] * \
                    self.beads[ind].binders[b].chemical_potential * \
                    self.mu_adjust_factor
                dE += states_ind[b] * \
                    self.beads[ind].binders[b].chemical_potential * \
                    self.mu_adjust_factor
        return dE

    def __str__(self):
        """Return string representation of the SSWLC.
        """
        return f"Polymer_Class<SSWLC>, {super(SSWLC, self).__str__()}"

    cpdef void _find_parameters(self, double[:] bead_length):
        """Look up elastic parameters of ssWLC for each bead_length.
        
        Interpolate from the parameter table to get the physical parameters of
        the WLC model matching the discretized WLC.
        
        Notes
        -----
        The units of the WLC parameters must be consistent with those in the 
        equation for elastic energy change. Lenght units are non-dimensionalized
        using the persistence length of the polymer.

        All parameters in the equation for elastic energy change are divided 
        or multiplied by `delta`. To avoid redundant calculations, we divide 
        or multiply the parameters by `delta` here and not each time elastic 
        energy change is calculated.
        
        The definition and units (after division by persistence length) of each 
        physical parameter are listed below:

            - delta         Distance btwn beads (Unitless)
            - eps_bend      Bending modulus (kbT)
            - gamma         Ground-state segment compression (Distance)
            - eps_par       Stretch Modulus (kbT / Distance**2)
            - eps_perp      Shear modulus (kbT / Distance**2)
            - eta           Bend-shear coupling (1 / Distance)

        Parameters
        ----------
        bead_length : double
            Dimensional distance between subsequent beads of the polymer (in nm)
        """
        cdef long i
        bead_length = np.asarray(bead_length)
        self.delta = np.zeros(len(bead_length))
        self.eps_bend = np.zeros(len(bead_length))
        self.gamma = np.zeros(len(bead_length))
        self.eps_par = np.zeros(len(bead_length))
        self.eps_perp = np.zeros(len(bead_length))
        self.eta = np.zeros(len(bead_length))
        for i in range(len(bead_length)):
            self.delta[i] = bead_length[i] / self.lp
            self.eps_bend[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 1]
            ) / self.delta[i]
            self.gamma[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 2]
            ) * self.delta[i] * self.lp
            self.eps_par[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 3]
            ) / (self.delta[i] * self.lp**2)
            self.eps_perp[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 4]
            ) / (self.delta[i] * self.lp**2)
            self.eta[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 5]
            ) / self.lp

    @classmethod
    def straight_line_in_x(
        cls, name: str, step_sizes: np.ndarray, **kwargs
    ):
        """Construct polymer initialized uniformly along the positve x-axis.

        Parameters
        ----------
        name : str
            Name of polymer being constructed
        step_sizes : np.ndarray (N-1,) of float
            Dimensional distance between subsequent beads of the polymer (in nm)
            represented as an (N-1,) array where N is the number of beads in the
            polymer.

        Returns
        -------
        SSWLC
            Object representation of a polymer currently configured as a
            straight line
        """
        # Get the cumulative sum of the step sizes
        csum = np.cumsum(step_sizes)
        # Initialize the positions and orientations of the polymer
        num_beads = len(step_sizes) + 1
        r = np.zeros((num_beads, 3))
        r[1:, 0] = csum.copy()
        t3 = np.zeros((num_beads, 3))
        t3[:, 0] = 1
        t2 = np.zeros((num_beads, 3))
        t2[:, 1] = 1
        return cls(name, r, t3=t3, t2=t2, bead_length=step_sizes, **kwargs)

    @classmethod
    def arbitrary_path_in_x_y(
        cls, name: str, num_beads: int, bead_length: float,
        shape_func: Callable[[float], float], step_size: float = 0.001,
        **kwargs
    ):
        """Construct a polymer initialized as y = f(x) from x = 0.

        Notes
        -----
        As is, this code does not support variable linker lengths.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : float
            Dimensional distance between subsequent beads of the polymer (in nm)
        shape_func : Callable[[float], float]
            Shape of the polymer where z = 0 and y = f(x)
        step_size : float
            Step size for numerical evaluation of contour length when
            building the polymer path

        Returns
        -------
        SSWLC
            Object representing a polymer following path y = f(x)
        """
        r = paths.coordinates_in_x_y(
            num_beads, bead_length, shape_func, step_size
        )
        t3, t2 = paths.get_tangent_vals_x_y(
            r[:, 0], shape_func, step_size
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

        Notes
        -----
        As is, this code does not support variable linker lengths.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : float
            Dimensional distance between subsequent beads of the polymer (in nm)
        shape_func_x : Callable[[float], float]
            Parametric functions to obtain the x coordinates of the path
        shape_func_y : Callable[[float], float]
            Parametric functions to obtain the y coordinates of the path
        shape_func_z : Callable[[float], float]
            Parametric functions to obtain the z coordinates of the path
        step_size : float
            Step size for numerical evaluation of contour length when
            building the polymer path

        Returns
        -------
        SSWLC
            Object representing a polymer following path x = X(t), y = Y(t),
            z = Z(t)
        """
        r, parameter_vals = paths.coordinates_in_x_y_z(
            num_beads, bead_length, shape_func_x, shape_func_y,
            shape_func_z, step_size
        )
        t3, t2 = paths.get_tangent_vals_x_y_z(
            parameter_vals, shape_func_x, shape_func_y, shape_func_z,
            step_size, r
        )
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def gaussian_walk_polymer(
        cls, name: str, num_beads: int, step_lengths: np.ndarray, **kwargs
    ):
        """Initialize a polymer to a Gaussian random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        step_lengths : np.ndarray (N-1,) float
            Dimensional distance between subsequent beads of the polymer (in nm)

        Returns
        -------
        SSWLC
            Object representing a polymer initialized as Gaussian random walk
        """
        r = paths.gaussian_walk(num_beads-1, step_lengths)
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=step_lengths, **kwargs)

    @classmethod
    def confined_gaussian_walk(
        cls, name: str, num_beads: int, step_lengths: np.ndarray,
        confine_type: str, confine_length: float, **kwargs
    ):
        """Initialize a polymer to a confined Gaussian random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        step_lengths : np.ndarray (N-1,) float
            Dimensional distance between subsequent beads of the polymer (in nm)
        confine_type : str
            Name of the confining boundary; to indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary. Length
            representation specified in function associated w/ `confine_type`

        Returns
        -------
        Polymer
            Object representing a polymer initialized as a confined Gaussian
            random walk
        """
        step_lengths = np.ascontiguousarray(step_lengths)
        r = paths.confined_gaussian_walk(
            num_beads, step_lengths, confine_type, confine_length
        )
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=step_lengths, **kwargs)

    @classmethod
    def confined_uniform_random_walk(
            cls, name: str, num_beads: int, step_lengths: float, confine_type: str,
            confine_length: float, **kwargs
    ):
        """Initialize a polymer to a confined uniform random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        step_lengths : float
            Dimensional distance between subsequent beads of the polymer (in nm)
        confine_type : str
            Name of the confining boundary; to indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary. Length
            representation specified in function associated w/ `confine_type`

        Returns
        -------
        Polymer
            Object representing a polymer initialized as a confined uniform
            random walk
        """
        r = paths.confined_uniform_random_walk(
            num_beads, step_lengths, confine_type, confine_length
        )
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=step_lengths, **kwargs)


cdef class Chromatin(SSWLC):
    """Stretchable, sharable WLC model of chromatin.

    Notes
    -----
    The persistence length of DNA is 53 nm. Each base pair of DNA is 0.34 nm
    along the double helix.

    Please see documentation for `PolymerBase` and `SSWLC` for parameter
    descriptions.

    TODO: This should be instantiated directly from SSWLC. This does not have
    the same level of abstraction as `SSTWLC`, which is also a subclass of
    `SSWLC`. Subclasses at the same level should have the same level of
    abstraction. To implement chromatin as an instance of the SSWLC, we need
    to adjust the `construct_beads` method to accept an arbitrary bead type.
    """

    def __init__(
        self,
        str name,
        np.ndarray[double, ndim=2] r,
        *,
        double[:] bead_length,
        double bead_rad = 5,
        np.ndarray[double, ndim=2] t3 = empty_2d,
        np.ndarray[double, ndim=2] t2 = empty_2d,
        np.ndarray[long, ndim=2] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        np.ndarray[long, ndim=2] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = "", long max_binders = -1, **kwargs
    ):
        """Construct a `Chromatin` fiber object as a subclass of `SSWLC`.

        Notes
        -----
        The persistence length of DNA is 53 nm. For now, this will be used as
        the persistence length of chromatin, stored in the `lp` attribute.

        Please see documentation for `PolymerBase.__init__()` and
        `SSWLC.__init__()` for parameter descriptions.
        """
        cdef double lp = 53
        super(Chromatin, self).__init__(
            name, r, bead_length=bead_length, bead_rad=bead_rad, lp=lp, t3=t3,
            t2=t2, states=states, binder_names=binder_names, log_path=log_path,
            chemical_mods=chemical_mods, chemical_mod_names=chemical_mod_names,
            max_binders=max_binders
        )
        self.construct_beads()

    cdef void construct_beads(self):
        """Construct Nucleosome objects forming beads of our Chromatin polymer.
        """
        self.beads = {
            i: beads.Nucleosome(
                id_=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                rad=self.bead_rad,
                states=self.states[i],
                binder_names=self.binder_names
            ) for i in range(self.r.shape[0])
        }


cdef class SSTWLC(SSWLC):
    """Class representation of stretchable, shearable wormlike chain w/ twist.

    Notes
    -----
    Please see documentation for `PolymerBase` and `SSWLC` for shared parameter
    descriptions.

    TODO: It does not make sense for Chromatin to be on the same level of
    abstraction as a SSTWLC.

    Parameters
    ----------
    lt : double
        Twist persistence length of the polymer (in nm)
    eps_twist : double
        Twist modulus of the polymer, divided by dimensionless `delta`,
        as appears in the equation for SSTWLC elastic energy (units of kbT)
    """

    def __init__(
        self,
        str name,
        double[:, ::1] r,
        *,
        double[:] bead_length,
        double lp,
        double lt,
        double bead_rad = 5,
        double[:, ::1] t3 = empty_2d,
        double[:, ::1] t2 = empty_2d,
        long[:, ::1] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        long[:, ::1] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = "", long max_binders = -1
    ):
        """Construct a `SSTWLC` polymer object as a subclass of `SSWLC`.

        Notes
        -----
        See documentation for `SSWLC` class additional parameter descriptions.

        Parameters
        ----------
        lt : double
            Twist persistence length (in nm)
        """
        cdef np.ndarray _arrays
        super(SSTWLC, self).__init__(
            name, r, t3=t3, t2=t2, states=states, binder_names=binder_names,
            log_path=log_path, chemical_mods=chemical_mods, lp=lp,
            bead_length=bead_length, chemical_mod_names=chemical_mod_names,
            max_binders=max_binders
        )
        self.bead_rad = bead_rad
        self.construct_beads()
        self.lt = lt
        self._find_parameters(self.bead_length)
        self.required_attrs = np.array([
            "name", "r", "t3", "t2", "states", "binder_names", "num_binders",
            "beads", "num_beads", "lp", "lt", "bead_rad"
        ])
        self._arrays = np.array(
            ['r', 't3', 't2', 'states', 'bead_length', 'chemical_mods']
        )
        self.check_attrs()

    cpdef void _find_parameters(self, double[:] step_lengths):
        """Look up elastic parameters of ssWLC for each bead_length.

        Notes
        -----
        See `_find_parameters` method in `SSWLC` class for additional
        documentation.
        
        A new physical parameters, the twist modulus `eps_twist`, is added to
        the SSTWLC model. Once divided by dimensionless `delta`, the twist 
        modulus carries units of kbT.

        TODO: Check definition of `delta`; check units of `eps_twist`.

        Parameters
        ----------
        step_lengths : np.ndarray (N-1,) of double
            Dimensional distance between subsequent beads of the polymer (in nm)
        """
        step_lengths = np.asarray(step_lengths)
        self.delta = np.zeros(len(step_lengths))
        self.eps_bend = np.zeros(len(step_lengths))
        self.gamma = np.zeros(len(step_lengths))
        self.eps_par = np.zeros(len(step_lengths))
        self.eps_perp = np.zeros(len(step_lengths))
        self.eta = np.zeros(len(step_lengths))
        self.eps_twist = np.zeros(len(step_lengths))
        for i in range(len(step_lengths)):
            self.delta[i] = step_lengths[i] / self.lp
            self.eps_bend[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 1]
            ) / self.delta[i]
            self.gamma[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 2]
            ) * self.delta[i] * self.lp
            self.eps_par[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 3]
            ) / (self.delta[i] * self.lp**2)
            self.eps_perp[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 4]
            ) / (self.delta[i] * self.lp**2)
            self.eta[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 5]
            ) / self.lp
            self.eps_twist[i] = self.lt / (self.delta[i] * self.lp)

    cdef double continuous_dE_poly(
        self,
        long ind0,
        long indf,
    ):
        """Compute change in polymer energy for a continuous bead region.

        Notes
        -----
        See documentation for `SSWLC.continuous_dE_poly()` class for details 
        and parameter/return descriptions.
        """
        cdef long ind0_m_1, indf_m_1
        cdef double delta_energy_poly

        ind0_m_1 = ind0 - 1
        indf_m_1 = indf - 1

        delta_energy_poly = 0
        if ind0 != 0:
            delta_energy_poly += self.bead_pair_dE_poly_forward_with_twist(
                self.r[ind0_m_1, :],
                self.r[ind0, :],
                self.r_trial[ind0, :],
                self.t3[ind0_m_1, :],
                self.t3[ind0, :],
                self.t3_trial[ind0, :],
                self.t2[ind0_m_1, :],
                self.t2[ind0, :],
                self.t2_trial[ind0, :],
                bond_ind = ind0_m_1
            )
        if indf != self.num_beads:
            delta_energy_poly += self.bead_pair_dE_poly_reverse_with_twist(
                self.r[indf_m_1, :],
                self.r_trial[indf_m_1, :],
                self.r[indf, :],
                self.t3[indf_m_1, :],
                self.t3_trial[indf_m_1, :],
                self.t3[indf, :],
                self.t2[indf_m_1, :],
                self.t2_trial[indf_m_1, :],
                self.t2[indf, :],
                bond_ind = indf_m_1
            )
        return delta_energy_poly

    cdef double E_pair_with_twist(
        self, double[:] bend, double dr_par, double[:] dr_perp, double omega,
        long bond_ind
    ):
        """Calculate elastic energy for a pair of beads.
        
        Notes
        -----
        See documentation for `SSWLC.E_pair()` class for more details.

        Parameters
        ----------
        bend : double[:]
            Bending vector
        dr_par : double
            Magnitude of the parallel component of the displacement vector
        dr_perp : double[:]
            Perpendicular component of the displacement vector
        omega : double
            Twist angle (in radians)
        bond_ind : long
            Index of the bond between the two beads

        Returns
        -------
        double
            Elastic energy of bond between the bead pair
        """
        cdef double E, delta_omega

        # Note, natural twist is based on the number of base pairs in a linker.
        # If the linker stretches, the mean-squared end-to-end distances of
        # polymer segments may deviate from theory, unless `bead_length[ind]`
        # is equal to the stretched bond length.

        # TODO: Adding twist seems to increase linker lengths! Why is this?

        # Adjust for natural twist of DNA (2 pi / 10.5 bp * 1 bp / 0.332 nm)
        delta_omega = omega - (
            self.bead_length[bond_ind] * NATURAL_TWIST_BARE / LENGTH_BP
        )
        # Assume polymer is well-behaved and twists < 2pi from relaxed state
        delta_omega -= 2 * np.pi * ((delta_omega + np.pi) // (2 * np.pi))
        assert -np.pi <= delta_omega < np.pi, \
            "Twist angle out of bounds."
        assert self.eps_twist[bond_ind] > 0, "Twist modulus must be positive."
        E = (
            0.5 * self.eps_bend[bond_ind] * vec_dot3(bend, bend) +
            0.5 * self.eps_par[bond_ind] * (dr_par - self.gamma[bond_ind])**2 +
            0.5 * self.eps_perp[bond_ind] * vec_dot3(dr_perp, dr_perp) +
            0.5 * self.eps_twist[bond_ind] * delta_omega**2
        )
        return E

    cdef double bead_pair_dE_poly_forward_with_twist(
        self,
        double[:] r_0,
        double[:] r_1,
        double[:] test_r_1,
        double[:] t3_0,
        double[:] t3_1,
        double[:] test_t3_1,
        double[:] t2_0,
        double[:] t2_1,
        double[:] test_t2_1,
        long bond_ind
    ):
        """Compute change in polymer energy when affecting a single bead pair.

        Notes
        -----
        See documentation for `SSWLC.bead_pair_dE_poly_forward()` class for 
        more details.

        TODO: Check sign of cross product (t2 x t3 OR t3 x t2). Store t1 
        arrays so they don't need to be recreated every time. Manually 
        implement dot products to speed up computation

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        r_1 : array_like (3,)
            Position vector of second bead in bend
        test_r_1 : array_like (3,)
            Position vector of second bead in bend (TRIAL MOVE)
        t3_0, t2_0 : array_like (3,)
            t3, t2 tangent vector of first bead in bend
        t3_1, t2_1 : array_like (3,)
            t3, t2 tangent vector of second bead in bend
        test_t3_1, test_t2_1 : array_like (3,)
            t3, t2 tangent vector of second bead in bend (TRIAL MOVE)
        bond_ind : long
            Index of the bond between the two beads

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef long i
        cdef double omega, omega_test, dr_par, dr_par_test
        omega = compute_twist_angle_omega(t2_0, t3_0, t2_1, t3_1)
        omega_test = compute_twist_angle_omega(
            t2_0, t3_0, test_t2_1, test_t3_1
        )
        for i in range(3):
            self.dr_test[i] = test_r_1[i] - r_0[i]
            self.dr[i] = r_1[i] - r_0[i]
        dr_par_test = vec_dot3(t3_0, self.dr_test)
        dr_par = vec_dot3(t3_0, self.dr)
        for i in range(3):
            self.dr_perp_test[i] = self.dr_test[i] - t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                test_t3_1[i] - t3_0[i] - self.dr_perp_test[i] *
                self.eta[bond_ind]
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[bond_ind]
            )
        return (self.E_pair_with_twist(
            self.bend_test, dr_par_test, self.dr_perp_test, omega_test, bond_ind
        ) - self.E_pair_with_twist(
            self.bend, dr_par, self.dr_perp, omega, bond_ind
        ))

    cdef double bead_pair_dE_poly_reverse_with_twist(
        self,
        double[:] r_0,
        double[:] test_r_0,
        double[:] r_1,
        double[:] t3_0,
        double[:] test_t3_0,
        double[:] t3_1,
        double[:] t2_0,
        double[:] test_t2_0,
        double[:] t2_1,
        long bond_ind
    ):
        """Compute change in polymer energy when affecting a single bead pair.

        Notes
        -----
        See documentation for `SSWLC.bead_pair_dE_poly_reverse()` class for 
        more details.

        TODO: Check sign of cross product (t2 x t3 OR t3 x t2). Store t1 
        arrays so they don't need to be recreated every time. Manually 
        implement dot products to speed up computation

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        test_r_0 : array_like (3,)
            Position vector of first bead in bend (TRIAL MOVE)
        r_1 : array_like (3,)
            Position vector of second bead in bend
        t3_0, t2_0 : array_like (3,)
            t3, t2 tangent vector of first bead in bend
        test_t3_0, test_t2_0 : array_like (3,)
            t3, t2 tangent vector of first bead in bend (TRIAL MOVE)
        t3_1, t2_1 : array_like (3,)
            t3, t2 tangent vector of second bead in bend
        bond_ind : long
            Index of the bond between the two beads

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef long i
        cdef double omega, omega_test, dr_par, dr_par_test
        omega = compute_twist_angle_omega(t2_0, t3_0, t2_1, t3_1)
        omega_test = compute_twist_angle_omega(
            test_t2_0, test_t3_0, t2_1, t3_1
        )
        for i in range(3):
            self.dr_test[i] = r_1[i]  - test_r_0[i]
            self.dr[i] = r_1[i] - r_0[i]
        dr_par_test = vec_dot3(test_t3_0, self.dr_test)
        dr_par = vec_dot3(t3_0, self.dr)
        for i in range(3):
            self.dr_perp_test[i] = self.dr_test[i] - test_t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                t3_1[i] - test_t3_0[i] - self.dr_perp_test[i] *
                self.eta[bond_ind]
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[bond_ind]
            )
        return (self.E_pair_with_twist(
            self.bend_test, dr_par_test, self.dr_perp_test, omega_test, bond_ind
        ) - self.E_pair_with_twist(
            self.bend, dr_par, self.dr_perp, omega, bond_ind
        ))

    cpdef double compute_E(self):
        """Compute the overall polymer energy at the current configuration.
        
        Notes
        -----
        This method loops over each bond in the polymer and calculates the 
        energy change of that bond.

        Returns
        -------
        double
            Configurational energy of the polymer.
        """
        cdef double E, dr_par
        cdef long i, j, i_m1
        cdef double[:] dr, dr_perp, bend
        cdef double[:] r_0, r_1, t3_0, t3_1, t2_0, t2_1

        E = 0
        for i in range(1, self.num_beads):
            i_m1 = i - 1
            r_0 = self.r[i_m1, :]
            r_1 = self.r[i, :]
            t3_0 = self.t3[i_m1, :]
            t3_1 = self.t3[i, :]
            t2_0 = self.t2[i_m1, :]
            t2_1 = self.t2[i, :]
            t1_0 = np.array([
                t2_0[1] * t3_0[2] - t2_0[2] * t3_0[1],
                t2_0[2] * t3_0[0] - t2_0[0] * t3_0[2],
                t2_0[0] * t3_0[1] - t2_0[1] * t3_0[0]
            ])
            t1_1 = np.array([
                t2_1[1] * t3_1[2] - t2_1[2] * t3_1[1],
                t2_1[2] * t3_1[0] - t2_1[0] * t3_1[2],
                t2_1[0] * t3_1[1] - t2_1[1] * t3_1[0]
            ])
            omega = np.arctan2(
                (np.dot(t2_0, t1_1) - np.dot(t1_0, t2_1)),
                (np.dot(t1_0, t1_1) + np.dot(t2_0, t2_1))
            )
            dr = vec_sub3(r_1, r_0)
            dr_par = vec_dot3(t3_0, dr)
            dr_perp = vec_sub3(dr, vec_scale3(t3_0, dr_par))
            bend = t3_1.copy()
            for j in range(3):
                bend[j] += -t3_0[j] - self.eta[i_m1] * dr_perp[j]
            E += self.E_pair_with_twist(bend, dr_par, dr_perp, omega, i_m1)
        return E

    cpdef double compute_E_no_twist(self):
        """Compute the overall polymer energy at the current configuration.
        
        Notes
        -----
        This method loops over each bond in the polymer and calculates the 
        energy change of that bond.

        Returns
        -------
        double
            Configurational energy of the polymer.
        """
        cdef double E, dr_par
        cdef long i, j, i_m1
        cdef double[:] dr, dr_perp, bend
        cdef double[:] r_0, r_1, t3_0, t3_1

        E = 0
        for i in range(1, self.num_beads):
            i_m1 = i - 1
            r_0 = self.r[i_m1, :]
            r_1 = self.r[i, :]
            t3_0 = self.t3[i_m1, :]
            t3_1 = self.t3[i, :]

            dr = vec_sub3(r_1, r_0)
            dr_par = vec_dot3(t3_0, dr)
            dr_perp = vec_sub3(dr, vec_scale3(t3_0, dr_par))
            bend = t3_1.copy()
            for j in range(3):
                bend[j] += -t3_0[j] - self.eta[i_m1] * dr_perp[j]
            E += self.E_pair(bend, dr_par, dr_perp, i_m1)
        return E


cdef class LoopedSSTWLC(SSTWLC):
    """Class representation of a looped SSTWLC.
    """

    def __init__(
        self,
        str name, double[:,::1] r, *, double bead_length, double lp=53,
        double lt=46.37, double bead_rad=5, double[:,::1] t3=empty_2d,
        double[:,::1] t2=empty_2d, long[:,::1] states=mty_2d_int,
        np.ndarray binder_names=empty_1d, long[:,::1] chemical_mods=mty_2d_int,
        np.ndarray chemical_mod_names=empty_1d, str log_path = "",
        long max_binders = -1
    ):
        """Construct a looped SSTWLC object as a subclass of `SSTWLC`.

        Notes
        -----
        See documentation for `SSTWLC` class for description of parameters.

        TODO: Change the default values of lt, lp to ones that makes more sense

        TODO: Implement variable linker lengths in this class!
        """
        super(LoopedSSTWLC, self).__init__(
            name, r, bead_length=bead_length, lp=lp, lt=lt, bead_rad=bead_rad,
            t3=t3, t2=t2, states=states, binder_names=binder_names,
            log_path=log_path, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names, max_binders=max_binders
        )

    @classmethod
    def looped_confined_gaussian_walk(
        cls,
        name: str,
        num_beads: int,
        bead_length: float,
        confine_type: str,
        confine_length: float,
        **kwargs
    ):
        """Initialize a polymer to a looped, confined Gaussian random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : float
            The amount of polymer path length between this bead and the next
            bead. For now, a constant value is assumed (the first value if an
            array is passed).
        confine_type : str
            Name of the confining boundary. To indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary. Length
            representation specified in function associated w/ `confine_type`

        Returns
        -------
        SSTWLC
            Object representing a SSTWLC initialized as a confined, looped
            Gaussian random walk
        """
        r = paths.looped_confined_gaussian_walk(
            num_beads, bead_length, confine_type, confine_length
        )
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    cdef void construct_beads(self):
        """Construct Nucleosome objects forming beads of our polymer.
        """
        self.beads = {
            i: beads.Nucleosome(
                id_=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                rad=self.bead_rad,
                states=self.states[i],
                binder_names=self.binder_names
            ) for i in range(self.r.shape[0])
        }

    cdef double continuous_dE_poly(
        self,
        long ind0,
        long indf,
    ):
        """Compute change in polymer energy for a continuous bead region.

        Notes
        -----
        See documentation for this method in `SSWLC` class for details and 
        parameter/return descriptions.
        """
        cdef long ind0_m_1, indf_m_1
        cdef double delta_energy_poly

        # Elastic energy change from LHS of the move
        delta_energy_poly = 0
        if ind0 != 0:
            ind0_m_1 = ind0 - 1
        else:
            ind0_m_1 = self.num_beads - 1
        delta_energy_poly += self.bead_pair_dE_poly_forward(
            self.r[ind0_m_1, :],
            self.r[ind0, :],
            self.r_trial[ind0, :],
            self.t3[ind0_m_1, :],
            self.t3[ind0, :],
            self.t3_trial[ind0, :],
            ind0_m_1
        )
        # Elastic energy change from RHS of the move
        indf_m_1 = indf - 1
        if indf == self.num_beads:
            indf = 0
        delta_energy_poly += self.bead_pair_dE_poly_reverse(
            self.r[indf_m_1, :],
            self.r_trial[indf_m_1, :],
            self.r[indf, :],
            self.t3[indf_m_1, :],
            self.t3_trial[indf_m_1, :],
            self.t3[indf, :],
            indf_m_1
        )

        return delta_energy_poly


cdef class DetailedChromatin(SSTWLC):
    """Class representation of a chromatin fiber with detailed nucleosomes.
    """
    def __init__(
        self,
        str name,
        double[:, ::1] r,
        *,
        double bp_wrap,
        double[:] bead_length,
        double lp,
        double lt,
        double[:, ::1] t3 = empty_2d,
        double[:, ::1] t2 = empty_2d,
        long[:, ::1] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        long[:, ::1] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = "", long max_binders = -1
    ):
        """Construct a detailed chromatin object as a subclass of `SSTWLC`.

        Notes
        -----
        See documentation for `SSTWLC` class for description of common
        parameters.

        Parameters
        ----------
        bp_wrap : double
            Number of base pairs wrapped around each nucleosome
        """
        self.bp_wrap = bp_wrap
        self.bead_rad = consts_dict["R"]
        super(DetailedChromatin, self).__init__(
            name, r, bead_length=bead_length, lp=lp, lt=lt,
            bead_rad=self.bead_rad, t3=t3, t2=t2, states=states,
            binder_names=binder_names, log_path=log_path,
            chemical_mods=chemical_mods, chemical_mod_names=chemical_mod_names,
            max_binders=max_binders
        )
        self.construct_beads()

    cdef void construct_beads(self):
        """Construct Nucleosome objects forming beads of our polymer.
        """
        self.beads = {
            i: beads.DetailedNucleosome(
                id_=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                bp_wrap=self.bp_wrap,
                states=self.states[i],
                binder_names=self.binder_names
            ) for i in range(self.r.shape[0])
        }

    cdef double continuous_dE_poly(
            self,
            long ind0,
            long indf,
    ):
        """Compute change in polymer energy for a continuous bead region.

        Notes
        -----
        See documentation for `SSWLC.continuous_dE_poly()` class for details 
        and parameter/return descriptions.
        """
        cdef long ind0_m_1, indf_m_1
        cdef double delta_energy_poly

        ind0_m_1 = ind0 - 1
        indf_m_1 = indf - 1

        delta_energy_poly = 0
        if ind0 != 0:
            # Compute the entry and exit positions and orientations of the
            # linker DNA for the first nucleosome in the continuous region
            ri_0, ro_0, t3i_0, t3o_0, t2i_0, t2o_0 =  \
                self.beads[ind0_m_1].update_configuration(
                    np.asarray(self.r[ind0_m_1, :]),
                    np.asarray(self.t3[ind0_m_1, :]),
                    np.asarray(self.t2[ind0_m_1, :])
                )
            ri_1_try, ro_1_try, t3i_1_try, t3o_1_try, t2i_1_try, t2o_1_try = \
                self.beads[ind0].update_configuration(
                    np.asarray(self.r_trial[ind0, :]),
                    np.asarray(self.t3_trial[ind0, :]),
                    np.asarray(self.t2_trial[ind0, :])
                )
            ri_1, ro_1, t3i_1, t3o_1, t2i_1, t2o_1 = \
                self.beads[ind0].update_configuration(
                    np.asarray(self.r[ind0, :]),
                    np.asarray(self.t3[ind0, :]),
                    np.asarray(self.t2[ind0, :])
                )

            # Check progress (can be removed later)
            # assert np.isclose(np.linalg.norm(t2i_0), 1)
            # assert np.isclose(np.linalg.norm(t2o_0), 1)
            # assert np.isclose(
            #     np.linalg.norm(ri_0 - np.asarray(self.r[ind0_m_1, :])),
            #     np.linalg.norm(ro_0 - np.asarray(self.r[ind0_m_1, :]))
            # ), "Error in calculation of entry and exit positions"
            # assert np.isclose(np.linalg.norm(t2i_1_try), 1)
            # assert np.isclose(np.linalg.norm(t2o_1_try), 1)
            # assert np.isclose(
            #     np.linalg.norm(ri_1_try - np.asarray(self.r_trial[ind0, :])),
            #     np.linalg.norm(ro_1_try - np.asarray(self.r_trial[ind0, :]))
            # ), "Error in calculation of entry and exit positions"
            # assert np.isclose(np.linalg.norm(t2i_1), 1)
            # assert np.isclose(np.linalg.norm(t2o_1), 1)
            # assert np.isclose(
            #     np.linalg.norm(ri_1 - np.asarray(self.r[ind0, :])),
            #     np.linalg.norm(ro_1 - np.asarray(self.r[ind0, :]))
            # ), "Error in calculation of entry and exit positions"

            delta_energy_poly += self.bead_pair_dE_poly_forward_with_twist(
                ro_0, ri_1, ri_1_try, t3o_0, t3i_1, t3i_1_try, t2o_0, t2i_1,
                t2i_1_try, bond_ind = ind0_m_1
            )
        if indf != self.num_beads:
            # Compute the entry and exit positions and orientations of the
            # linker DNA for the last nucleosome in the continuous region
            ri_0_try, ro_0_try, t3i_0_try, t3o_0_try, t2i_0_try, t2o_0_try = \
                self.beads[indf_m_1].update_configuration(
                    np.asarray(self.r_trial[indf_m_1, :]),
                    np.asarray(self.t3_trial[indf_m_1, :]),
                    np.asarray(self.t2_trial[indf_m_1, :])
                )
            ri_0, ro_0, t3i_0, t3o_0, t2i_0, t2o_0 = \
                self.beads[indf_m_1].update_configuration(
                    np.asarray(self.r[indf_m_1, :]),
                    np.asarray(self.t3[indf_m_1, :]),
                    np.asarray(self.t2[indf_m_1, :])
                )
            ri_1, ro_1, t3i_1, t3o_1, t2i_1, t2o_1 = \
                self.beads[indf].update_configuration(
                    np.asarray(self.r[indf, :]),
                    np.asarray(self.t3[indf, :]),
                    np.asarray(self.t2[indf, :])
                )

            # Check progress (can be removed later)
            # assert np.isclose(np.linalg.norm(t2i_0_try), 1)
            # assert np.isclose(np.linalg.norm(t2o_0_try), 1)
            # assert np.isclose(
            #     np.linalg.norm(ri_0_try - np.asarray(self.r_trial[indf_m_1,:])),
            #     np.linalg.norm(ro_0_try - np.asarray(self.r_trial[indf_m_1,:]))
            # ), "Error in calculation of entry and exit positions"
            # assert np.isclose(np.linalg.norm(t2i_0), 1)
            # assert np.isclose(np.linalg.norm(t2o_0), 1)
            # assert np.isclose(
            #     np.linalg.norm(ri_0 - np.asarray(self.r[indf_m_1, :])),
            #     np.linalg.norm(ro_0 - np.asarray(self.r[indf_m_1, :]))
            # ), "Error in calculation of entry and exit positions"
            # assert np.isclose(np.linalg.norm(t2i_1), 1)
            # assert np.isclose(np.linalg.norm(t2o_1), 1)
            # assert np.isclose(
            #     np.linalg.norm(ri_1 - np.asarray(self.r[indf, :])),
            #     np.linalg.norm(ro_1 - np.asarray(self.r[indf, :]))
            # ), "Error in calculation of entry and exit positions"

            delta_energy_poly += self.bead_pair_dE_poly_reverse_with_twist(
                ro_0, ro_0_try, ri_1, t3o_0, t3o_0_try, t3i_1, t2o_0, t2o_0_try,
                t2i_1, bond_ind=indf_m_1
            )
        return delta_energy_poly


cdef class DetailedChromatin2(DetailedChromatin):
    """Chromatin fiber with detailed nucleosomes without nucleosome diameter.

    Notes
    -----
    This class is identical to `DetailedChromatin` except that the nucleosome
    diameter is not included in the energy calculations. This is useful for
    comparison with theoretical end-to-end distances of a kinked wormlike chain.

    Parameters
    ----------
    See documentation for `DetailedChromatin` class for details and parameter
    descriptions.
    """
    def __init__(
        self,
        str name,
        double[:, ::1] r,
        *,
        double bp_wrap,
        double[:] bead_length,
        double lp,
        double lt,
        double[:, ::1] t3 = empty_2d,
        double[:, ::1] t2 = empty_2d,
        long[:, ::1] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        long[:, ::1] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = "", long max_binders = -1
    ):
        """Initialize a detailed chromatin fiber.

        Parameters
        ----------
        see `DetailedChromatin.__init__()` for details
        """
        super().__init__(
            name, r, bp_wrap=bp_wrap, bead_length=bead_length,
            lp=lp, lt=lt, t3=t3, t2=t2, states=states,
            binder_names=binder_names, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names, log_path=log_path,
            max_binders=max_binders
        )

    cdef double continuous_dE_poly(
            self,
            long ind0,
            long indf,
    ):
        """Compute change in polymer energy for a continuous bead region.

        Notes
        -----
        See documentation for `SSWLC.continuous_dE_poly()` class for details 
        and parameter/return descriptions.
        """
        cdef long ind0_m_1, indf_m_1
        cdef double delta_energy_poly

        ind0_m_1 = ind0 - 1
        indf_m_1 = indf - 1

        delta_energy_poly = 0
        if ind0 != 0:
            # Compute the entry and exit positions and orientations of the
            # linker DNA for the first nucleosome in the continuous region
            _, _, t3i_0, t3o_0, t2i_0, t2o_0 = \
                self.beads[ind0_m_1].update_configuration(
                    self.r[ind0_m_1, :], self.t3[ind0_m_1, :],
                    self.t2[ind0_m_1, :]
                )
            _, _, t3i_1, t3o_1, t2i_1, t2o_1 = \
                self.beads[ind0].update_configuration(
                    self.r[ind0, :], self.t3[ind0, :], self.t2[ind0, :]
                )
            _, _, t3i_1_try, t3o_1_try, t2i_1_try, t2o_1_try = \
                self.beads[ind0].update_configuration(
                    self.r_trial[ind0, :], self.t3_trial[ind0, :],
                    self.t2_trial[ind0, :]
                )
            delta_energy_poly += self.bead_pair_dE_poly_forward_with_twist(
                self.r[ind0_m_1, :], self.r[ind0, :],  self.r_trial[ind0, :],
                t3o_0, t3i_1, t3i_1_try, t2o_0, t2i_1, t2i_1_try,
                bond_ind = ind0_m_1
            )
        if indf != self.num_beads:
            # Compute the entry and exit positions and orientations of the
            # linker DNA for the last nucleosome in the continuous region
            _, _, t3i_0, t3o_0, t2i_0, t2o_0 = \
                self.beads[indf_m_1].update_configuration(
                    self.r[indf_m_1, :], self.t3[indf_m_1, :],
                    self.t2[indf_m_1, :]
                )
            _, _, t3i_1, t3o_1, t2i_1, t2o_1 = \
                self.beads[indf].update_configuration(
                    self.r[indf, :], self.t3[indf, :], self.t2[indf, :]
                )
            _, _, t3i_0_try, t3o_0_try, t2i_0_try, t2o_0_try = \
                self.beads[indf_m_1].update_configuration(
                    self.r_trial[indf_m_1, :], self.t3_trial[indf_m_1, :],
                    self.t2_trial[indf_m_1, :]
                )
            delta_energy_poly += self.bead_pair_dE_poly_reverse_with_twist(
                self.r[indf_m_1, :], self.r_trial[indf_m_1, :], self.r[indf, :],
                t3o_0, t3o_0_try, t3i_1, t2o_0, t2o_0_try, t2i_1,
                bond_ind = indf_m_1
            )
        return delta_energy_poly


cdef class DetailedChromatinWithSterics(DetailedChromatin):
    """Chromatin fiber with detailed nucleosomes and steric interactions.

    Notes
    -----
    Because we are computing pairwise distances explicitly, we can evaluate
    binder protein interactions specifically. For this reason, we do not need
    to use a field. The field is never active. We can specify a null field as
    a place-holder. Beware that the field will not contribute to the energy
    associated with an instance of this class.
    """
    def __init__(
        self,
        str name,
        double[:, ::1] r,
        *,
        double bp_wrap,
        double[:] bead_length,
        double lp,
        double lt,
        binders = None,
        double[:, ::1] t3 = empty_2d,
        double[:, ::1] t2 = empty_2d,
        long[:, ::1] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        long[:, ::1] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = "", long max_binders = -1
    ):
        """Initialize a detailed chromatin fiber with steric interactions.

        Parameters
        ----------
        see `DetailedChromatin.__init__()` for details

        binders : sequence of Binder objects, optional
            Representation of the interacting binders on the chromatin fiber.
            If None, no binders are present. Default is None.
        """
        super(DetailedChromatinWithSterics, self).__init__(
            name, r, bp_wrap=bp_wrap, bead_length=bead_length,
            lp=lp, lt=lt, t3=t3, t2=t2, states=states,
            binder_names=binder_names, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names, log_path=log_path,
            max_binders=max_binders
        )
        self.V0 = 1.0
        self.excluded_distance = self.beads[0].rad * 2.0
        self.distances = np.zeros((self.num_beads, self.num_beads))
        self.distances_trial = np.zeros((self.num_beads, self.num_beads))
        self.get_distances()

        # Load binders
        self.binders = binders
        if self.binders is not None:
            self.num_binders = len(self.binders)
        else:
            self.num_binders = 0

    cpdef bint is_field_active(self):
        """The field is never active for this class.
        
        Notes
        -----
        All interactions are handled using explicit pairwise interactions.
        """
        return 0

    cpdef void get_distances(self):
        """Get the distances between all pairs of nucleosomes.
        """
        cdef long i, j
        for i in range(self.num_beads - 1):
            self.distances[i, i] = 0.0
            self.distances_trial[i, i] = 0.0
            for j in range(i + 1, self.num_beads):
                self.distances[i, j] = np.sqrt(
                    (self.r[i, 0] - self.r[j, 0]) ** 2 + \
                    (self.r[i, 1] - self.r[j, 1]) ** 2 + \
                    (self.r[i, 2] - self.r[j, 2]) ** 2
                )
                self.distances_trial[i, j] = np.sqrt(
                    (self.r_trial[i, 0] - self.r_trial[j, 0]) ** 2 + \
                    (self.r_trial[i, 1] - self.r_trial[j, 1]) ** 2 + \
                    (self.r_trial[i, 2] - self.r_trial[j, 2]) ** 2
                )
                self.distances[j, i] = self.distances[i, j]
                self.distances_trial[j, i] = self.distances_trial[i, j]

    cpdef void get_delta_distances(self, long ind0, long indf):
        """Get changes in distances, rather than every distance
        
        Notes
        -----
        This function operates a lot like `get_distances`, but only computes
        the pairwise distances only between beads affected by a move and all
        other beads in the system. This is useful for computing the change in
        energy associated with a proposed move in a more efficient manner.
        
        This method only updates the `distances_trial` attribute of the class.
        
        Parameters
        ----------
        ind0, indf : long
            Indices of the first and one-past-the-last beads affected by the
            move
        """
        cdef long i, j
        for i in range(ind0, indf):
            self.distances_trial[i, i] = 0.0
            for j in range(self.num_beads):

                # Let's avoid unnecessary computation
                if i == j:
                    continue

                self.distances_trial[i, j] = np.sqrt(
                    (self.r_trial[i, 0] - self.r_trial[j, 0]) ** 2 + \
                    (self.r_trial[i, 1] - self.r_trial[j, 1]) ** 2 + \
                    (self.r_trial[i, 2] - self.r_trial[j, 2]) ** 2
                )
                self.distances_trial[j, i] = self.distances_trial[i, j]

    cpdef double eval_delta_steric_clashes(self, long ind0, long indf):
        """Get the change in steric clashes associated with a move.
        
        Notes
        -----
        This method computes the steric clashes, but does so only by checking
        the beads that were affected by an MC move. There is not need to
        re-check for steric clashes between unmoved beads.
        
        Returns
        -------
        double
            Change in the number of steric clashes associated with the move.
        """
        cdef long i, j, num_beads
        cdef double overlap_frac, E_clash
        E_clash = 0
        num_beads = self.num_beads
        for i in range(ind0, indf):
            for j in range(num_beads):

                # Don't double count!
                if j >= i and (ind0 <= j < indf):
                    continue

                if self.distances_trial[i, j] < self.excluded_distance:
                    overlap_frac = (
                        self.excluded_distance / self.distances_trial[i, j]
                    )
                    E_clash += self.V0 * (overlap_frac**12 -
                        2 * overlap_frac**6 + 1)
                if self.distances[i, j] < self.excluded_distance:
                    overlap_frac = (
                        self.excluded_distance / self.distances[i, j]
                    )
                    E_clash -= self.V0 * (overlap_frac**12 -
                        2 * overlap_frac**6 + 1)
        return E_clash

    cpdef double eval_E_steric_clashes(self):
        """Compute the Energy associated with steric clashes.
        
        Returns
        -------
        double
            Energy associated with steric clashes in the current configuration.
        """
        cdef long i, j, num_beads
        cdef double overlap_frac
        E_clash = 0
        num_beads = len(self.distances)
        for i in range(0, num_beads - 1):
            for j in range(i + 1, num_beads):
                if self.distances[i, j] < self.excluded_distance:
                    overlap_frac = (
                        self.excluded_distance / self.distances[i, j]
                    )
                    E_clash += self.V0 * (overlap_frac ** 12 -
                                          2 * overlap_frac ** 6 + 1)
        return E_clash

    cpdef long check_steric_clashes(self, double[:, ::1] distances):
        """Check for steric clashes between nucleosomes.

        Returns
        -------
        long
            Number of Steric clashes.
        """
        cdef long i, j, n_clashes, num_beads
        n_clashes = 0
        num_beads = len(distances)
        for i in range(num_beads - 1):
            for j in range(i + 1, num_beads):
                if distances[i, j] < self.excluded_distance:
                    n_clashes += 1
        return n_clashes

    cpdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    ):
        """Compute change in energy for proposed transformation.
        
        Notes
        -----
        This method accounts for the elastic energy of the polymer, the steric
        interactions between nucleosomes, and the interactions between binders.
        
        Parameters
        ----------
        move_name : str
            Name of the MC move for which the energy change is being calculated
        inds : array_like (N,)
            Indices of N beads affected by the MC move
        n_inds : long
            Number of beads affected by the MC move

        Returns
        -------
        delta_energy_poly : double
            Change in polymer elastic energy associated with the trial move
        """
        cdef double delta_energy_poly, dE_clashes
        cdef long ind0, indf, ind, i

        delta_energy_poly = 0

        # Check Sterics: Compute pairwise distances between nucleosomes
        # New steric clashes only occur for moves involving translations
        if (
            move_name == "slide" or move_name == "end_pivot" or
            move_name == "crank_shaft"
        ):
            ind0 = inds[0]
            indf = inds[n_inds-1] + 1
            self.get_delta_distances(ind0, indf)
            dE_clashes = self.eval_delta_steric_clashes(ind0, indf)
            delta_energy_poly += dE_clashes

        if move_name == "change_binding_state":
            if self.num_binders > 0:
                ind0 = inds[0]
                indf = inds[n_inds-1] + 1
                delta_energy_poly += self.binding_dE(ind0, indf, n_inds)
                # Compute change in reader protein interactions
                delta_energy_poly += self.evaluate_binder_interactions()
        elif (
            move_name == "slide" or move_name == "end_pivot" or
            move_name == "crank_shaft"
        ):
            ind0 = inds[0]
            indf = inds[n_inds-1] + 1
            delta_energy_poly += self.continuous_dE_poly(ind0, indf)
        elif move_name == "tangent_rotation":
            for i in range(n_inds):
                ind = inds[i]
                ind0 = ind
                indf = ind + 1
                delta_energy_poly += self.continuous_dE_poly(ind0, indf)

        return delta_energy_poly

    cpdef double evaluate_binder_interactions(self):
        """Evaluate change in energy associated with binding states.

        Notes
        -----
        Pairwise distances should be updated before running this method.
        To avoid double counting cross-interactions, the cross-interaction
        energy associated with two binders should only be listed in the
        properties of one of the binders. The cross-interaction energy should
        be listed as 0 for the pair in the properties of the other binder.

        Returns
        -------
        double
            Change in energy associated with binding states and reader protein
            interactions.
        """
        cdef long i, j, k, l
        cdef double delta_energy_binding = 0.0
        cdef double n_pairs, n_pairs_trial, cutoff_distance

        # Self interactions, same nucleosome
        for i in range(self.num_binders):
            for j in range(self.num_beads):
                n_pairs = self.states[j, i] * (self.states[j, i] - 1) / 2
                n_pairs_trial = \
                    self.states_trial[j, i] * (self.states_trial[j, i] - 1) / 2
                delta_energy_binding += self.binders[i].interaction_energy * (
                    n_pairs_trial - n_pairs
                )
        # Self interaction, different nucleosomes
        for i in range(self.num_binders):
            for j in range(self.num_beads - 1):
                for k in range(j + 1, self.num_beads):
                    cutoff_distance = self.binders[i].interaction_radius
                    if self.distances_trial[j, k] <= cutoff_distance:
                        n_pairs_trial = \
                            self.states_trial[j, i] * self.states_trial[k, i]
                        delta_energy_binding += \
                            self.binders[i].interaction_energy * n_pairs_trial
                    if self.distances[j, k] <= cutoff_distance:
                        n_pairs = self.states[j, i] * self.states[k, i]
                        delta_energy_binding -= \
                            self.binders[i].interaction_energy * n_pairs
        # Cross-interaction, same nucleosome
        for i in range(self.num_binders):
            for j in range(self.num_binders):
                if j == i:
                    continue
                for k in range(self.num_beads):
                    n_pairs = self.states[k, i] * self.states[k, j]
                    n_pairs_trial = \
                        self.states_trial[k, i] * self.states_trial[k, j]
                    delta_energy_binding += \
                        self.binders[i].cross_talk_interaction_energy[
                            self.binders[j].name
                        ] * (
                            n_pairs_trial - n_pairs
                        )
        # Cross-interaction, different nucleosomes
        for i in range(self.num_binders):
            for j in range(self.num_binders):
                if j == i:
                    continue
                for k in range(self.num_beads - 1):
                    for l in range(k + 1, self.num_beads):
                        cutoff_distance = self.binders[i].interaction_radius
                        if self.distances_trial[k, l] <= cutoff_distance:
                            n_pairs_trial = \
                                self.states_trial[k, i]*self.states_trial[l, j]
                            delta_energy_binding += \
                                self.binders[i].cross_talk_interaction_energy[
                                    self.binders[j].name
                                ] * n_pairs_trial
                        if self.distances[k, l] <= cutoff_distance:
                            n_pairs = self.states[k, i] * self.states[l, j]
                            delta_energy_binding -= \
                                self.binders[i].cross_talk_interaction_energy[
                                    self.binders[j].name
                                ] * n_pairs
        return delta_energy_binding

    cpdef double get_E_bind(self):
        """Evaluate energy associated with current binding states.

        Notes
        -----
        Pairwise distances should be updated before running this method.
        To avoid double counting cross-interactions, the cross-interaction
        energy associated with two binders should only be listed in the
        properties of one of the binders. The cross-interaction energy should
        be listed as 0 for the pair in the properties of the other binder.

        Returns
        -------
        double
            Change in energy associated with binding states and reader protein
            interactions.
        """
        cdef long i, j, k, l
        cdef double delta_energy_binding = 0.0
        cdef double n_pairs, n_pairs_trial, cutoff_distance

        # Self interactions, same nucleosome
        for i in range(self.num_binders):
            for j in range(self.num_beads):
                n_pairs = self.states[j, i] * (self.states[j, i] - 1) / 2
                delta_energy_binding += \
                    self.binders[i].interaction_energy * n_pairs
        # Self interaction, different nucleosomes
        for i in range(self.num_binders):
            for j in range(self.num_beads - 1):
                for k in range(j + 1, self.num_beads):
                    cutoff_distance = self.binders[i].interaction_radius
                    if self.distances[j, k] <= cutoff_distance:
                        n_pairs = self.states[j, i] * self.states[k, i]
                        delta_energy_binding += \
                            self.binders[i].interaction_energy * n_pairs
        # Cross-interaction, same nucleosome
        for i in range(self.num_binders):
            for j in range(self.num_binders):
                if j == i:
                    continue
                for k in range(self.num_beads):
                    n_pairs = self.states[k, i] * self.states[k, j]
                    delta_energy_binding += \
                        self.binders[i].cross_talk_interaction_energy[
                            self.binders[j].name
                        ] * n_pairs
        # Cross-interaction, different nucleosomes
        for i in range(self.num_binders):
            for j in range(self.num_binders):
                if j == i:
                    continue
                for k in range(self.num_beads - 1):
                    for l in range(k + 1, self.num_beads):
                        cutoff_distance = self.binders[i].interaction_radius
                        if self.distances[k, l] <= cutoff_distance:
                            n_pairs = self.states[k, i] * self.states[l, j]
                            delta_energy_binding += \
                                self.binders[i].cross_talk_interaction_energy[
                                    self.binders[j].name
                                ] * n_pairs
        return delta_energy_binding

    cpdef double compute_E(self):
        """Compute the overall polymer energy at the current configuration.
        
        Notes
        -----
        This method loops over each bond in the polymer and calculates the 
        energy change of that bond.

        Returns
        -------
        double
            Configurational energy of the polymer.
        """
        cdef double E, dr_par
        cdef long i, j, i_m1
        cdef double[:] dr, dr_perp, bend
        cdef double[:] r_0, r_1, t3_0, t3_1, t2_0, t2_1

        E = 0
        for i in range(1, self.num_beads):
            i_m1 = i - 1

            # Get the entry/exit positions and orientations!
            ri_0, ro_0, t3i_0, t3o_0, t2i_0, t2o_0 = \
                self.beads[i_m1].update_configuration(
                    np.asarray(self.r[i_m1, :]),
                    np.asarray(self.t3[i_m1, :]),
                    np.asarray(self.t2[i_m1, :])
                )
            ri_1, ro_1, t3i_1, t3o_1, t2i_1, t2o_1 = \
                self.beads[i].update_configuration(
                    np.asarray(self.r[i, :]),
                    np.asarray(self.t3[i, :]),
                    np.asarray(self.t2[i, :])
                )
            t1_0 = np.array([
                t2o_0[1] * t3o_0[2] - t2o_0[2] * t3o_0[1],
                t2o_0[2] * t3o_0[0] - t2o_0[0] * t3o_0[2],
                t2o_0[0] * t3o_0[1] - t2o_0[1] * t3o_0[0]
            ])
            t1_1 = np.array([
                t2i_1[1] * t3i_1[2] - t2i_1[2] * t3i_1[1],
                t2i_1[2] * t3i_1[0] - t2i_1[0] * t3i_1[2],
                t2i_1[0] * t3i_1[1] - t2i_1[1] * t3i_1[0]
            ])
            omega = np.arctan2(
                (np.dot(t2o_0, t1_1) - np.dot(t1_0, t2i_1)),
                (np.dot(t1_0, t1_1) + np.dot(t2o_0, t2i_1))
            )
            dr = vec_sub3(ri_1, ro_0)
            dr_par = vec_dot3(t3o_0, dr)
            dr_perp = vec_sub3(dr, vec_scale3(t3o_0, dr_par))
            bend = t3i_1.copy()
            for j in range(3):
                bend[j] += -t3o_0[j] - self.eta[i_m1] * dr_perp[j]
            E += self.E_pair_with_twist(bend, dr_par, dr_perp, omega, i_m1)
        E_elastic = E
        E_elastic = E

        # Compute the energy change associated with sterics
        E = 0
        # Compute pairwise distances between nucleosomes
        self.get_distances()
        # Evaluate the energy of steric clashes
        E = self.eval_E_steric_clashes()
        E_sterics = E

        # Compute change in reader protein interactions
        E = 0
        if self.num_binders > 0:
            E = self.get_E_bind()
        E_interactions = E

        # compute the total energy
        E = E_elastic + E_sterics + E_interactions

        return E

    cpdef dict compute_E_detailed(self):
        """Compute the overall polymer energy at the current configuration.
        
        Notes
        -----
        This method loops over each bond in the polymer and calculates the 
        energy change of that bond. The method returns the energy as a
        dictionary with the elastic, steric, and interaction components of
        the total energy.

        Returns
        -------
        Dict[str, float]
            Dictionary of the elastic, steric, interaction, and total energy
            of a configuration.
        """
        cdef double E, dr_par
        cdef long i, j, i_m1
        cdef double[:] dr, dr_perp, bend
        cdef double[:] r_0, r_1, t3_0, t3_1, t2_0, t2_1

        E = 0
        for i in range(1, self.num_beads):
            i_m1 = i - 1

            # Get the entry/exit positions and orientations!
            ri_0, ro_0, t3i_0, t3o_0, t2i_0, t2o_0 = \
                self.beads[i_m1].update_configuration(
                    np.asarray(self.r[i_m1, :]),
                    np.asarray(self.t3[i_m1, :]),
                    np.asarray(self.t2[i_m1, :])
                )
            ri_1, ro_1, t3i_1, t3o_1, t2i_1, t2o_1 = \
                self.beads[i].update_configuration(
                    np.asarray(self.r[i, :]),
                    np.asarray(self.t3[i, :]),
                    np.asarray(self.t2[i, :])
                )
            t1_0 = np.array([
                t2o_0[1] * t3o_0[2] - t2o_0[2] * t3o_0[1],
                t2o_0[2] * t3o_0[0] - t2o_0[0] * t3o_0[2],
                t2o_0[0] * t3o_0[1] - t2o_0[1] * t3o_0[0]
            ])
            t1_1 = np.array([
                t2i_1[1] * t3i_1[2] - t2i_1[2] * t3i_1[1],
                t2i_1[2] * t3i_1[0] - t2i_1[0] * t3i_1[2],
                t2i_1[0] * t3i_1[1] - t2i_1[1] * t3i_1[0]
            ])

            # Forward direction
            omega = np.arctan2(
                (np.dot(t2o_0, t1_1) - np.dot(t1_0, t2i_1)),
                (np.dot(t1_0, t1_1) + np.dot(t2o_0, t2i_1))
            )
            dr = vec_sub3(ri_1, ro_0)
            dr_par = vec_dot3(t3o_0, dr)
            dr_perp = vec_sub3(dr, vec_scale3(t3o_0, dr_par))
            bend = t3i_1.copy()
            for j in range(3):
                bend[j] += -t3o_0[j] - self.eta[i_m1] * dr_perp[j]
            E += self.E_pair_with_twist(bend, dr_par, dr_perp, omega, i_m1)

            # Reverse Direction
            omega = np.arctan2(
                (np.dot(t2i_1, t1_0) - np.dot(t1_1, t2o_0)),
                (np.dot(t1_1, t1_0) + np.dot(t2i_1, t2o_0))
            )
            dr = vec_sub3(ro_0, ri_1)
            dr_par = vec_dot3(t3i_1, dr)
            dr_perp = vec_sub3(dr, vec_scale3(t3i_1, dr_par))
            bend = t3o_0.copy()
            for j in range(3):
                bend[j] += -t3i_1[j] - self.eta[i_m1] * dr_perp[j]
            E += self.E_pair_with_twist(bend, dr_par, dr_perp, omega, i_m1)

        E_elastic = E

        # Compute the energy change associated with sterics
        E = 0
        # Compute pairwise distances between nucleosomes
        self.get_distances()
        # Evaluate the energy of steric clashes
        E = self.eval_E_steric_clashes()
        E_sterics = E

        # Compute change in reader protein interactions
        E = 0
        if self.num_binders > 0:
            E = self.get_E_bind()
        E_interactions = E

        # compute the total energy
        E = E_elastic + E_sterics + E_interactions

        return {
            "elastic": E_elastic, "steric": E_sterics,
            "interaction": E_interactions, "total": E
        }


cpdef double sin_func(double x):
    """Sine function to which the polymer will be initialized.

    Parameters
    ----------
    x : double
        Input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    return 50 * sin(x / 35)


cpdef double helix_parametric_x(double t):
    """Parametric equation for x-coordinates of a helix.

    Parameters
    ----------
    t : double
        Parameter input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    cdef double x = 60 * cos(t)
    return x


cpdef double helix_parametric_y(double t):
    """Parametric equation for y-coordinates of a helix.

    Parameters
    ----------
    t : double
        Parameter input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    cdef double y = 60 * sin(t)
    return y


cpdef double helix_parametric_z(double t):
    """Parametric equation for z-coordinates of a helix.

    Parameters
    ----------
    t : double
        Parameter input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    cdef double z = 20 * t
    return z


cpdef double compute_twist_angle_omega(
    double[:] t2_0, double[:] t3_0, double[:] t2_1, double[:] t3_1
):
    """Compute twist angle between two sets of orientation vectors.
    
    Parameters
    ----------
    t2_0, t3_0 : double[:] of shape (3,)
        T2 and T3 orientation vectors of the first bead
    t2_1, t3_1 : double[:] of shape (3,)
        T2 and T3 orientation vectors of the second bead
    
    Returns
    -------
    double
        Twist angle (in radians)
    """
    cdef double[:] t1_0, t1_1
    cdef double omega

    t1_0 = np.array([
        t2_0[1] * t3_0[2] - t2_0[2] * t3_0[1],
        t2_0[2] * t3_0[0] - t2_0[0] * t3_0[2],
        t2_0[0] * t3_0[1] - t2_0[1] * t3_0[0]
    ])
    t1_1 = np.array([
        t2_1[1] * t3_1[2] - t2_1[2] * t3_1[1],
        t2_1[2] * t3_1[0] - t2_1[0] * t3_1[2],
        t2_1[0] * t3_1[1] - t2_1[1] * t3_1[0]
    ])
    omega = np.arctan2(
        (np.dot(t2_0, t1_1) - np.dot(t1_0, t2_1)),
        (np.dot(t1_0, t1_1) + np.dot(t2_0, t2_1))
    )
    return omega
