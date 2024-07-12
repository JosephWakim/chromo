"""Beads represent monomeric units forming the polymer.
"""

# Built-in Modules
from abc import ABC, abstractmethod
from typing import (Sequence, Optional, List)

# External Modules
import numpy as np

# Custom Modules
from .util.nucleo_geom import \
    get_exiting_orientations, LENGTH_BP, consts_dict, get_r
from .binders import Binder, get_by_name
from .util import linalg as la
from .util.gjk import gjk_collision


class Bead(ABC):
    """Abstract class representation of a bead of a polymer.

    Attributes
    ----------
    id : int
        Bead identifier; typically the beads index position in the chain
    r : array_like (3,) of double
        Position of the bead
    t3 : array_like (3,) of double
        Tangential orientation vector defining the bead
    t2 : array_like (3,) of double
        Tangential orientation vector defining the bead; orthogonal to `t3`
    states : array_like (M,) of int
        State of each bound protein on the bead
    binder_names : array_like (M,) of str
        Name of each bound protein on the bead
    """

    def __init__(
        self, id_: int, r: np.ndarray, t3: Optional[np.ndarray] = None,
        t2: Optional[np.ndarray] = None, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize the bead object.

        Notes
        -----
        These are the required attributes that any type of bead must have for
        proper compatibility with the rest of the simulator.

        Parameters
        ----------
        id_ : int
            Bead identifier
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : Optional np.ndarray (3,)
            Tangent vector defining orientation of nucleosome, default = `None`
        t2 : Optional np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`, default = `None`
        states : Optional[np.ndarray] (M, ) of int
            State of each of the M binders being tracked, default = None
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        """
        self.id = id_
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.states = states
        self.binder_names = binder_names
        if binder_names is not None:
            self.binders: Optional[List[Binder]] =\
                [get_by_name(name) for name in binder_names]
        else:
            self.binders = None

    @abstractmethod
    def test_collision(self, **kwargs):
        """Test collisions with another bead or a confinement.
        """
        pass

    @abstractmethod
    def print_properties(self):
        """Print properties of the bead.
        """
        pass


class GhostBead(Bead):
    """Class representation of a "ghost bead" for which collisions are ignored.

    Notes
    -----
    See documentation for `Bead` class for additional attribute definitions.

    Attributes
    ----------
    rad : double
        Radius of the spherical ghost bead
    vol : double
        Volume of the spherical ghost bead
    kwargs : Dict
        Additional named properties of the ghost bead (for later use)
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: Optional[np.ndarray] = None,
        t2: Optional[np.ndarray] = None, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None, rad: Optional[float] = 5,
        **kwargs
    ):
        """Initialize the non-interacting ghost bead object.

        Parameters
        ----------
        id_ : int
            Bead identifier
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : Optional np.ndarray (3,)
            Tangent vector defining orientation of nucleosome, default = `None`
        t2 : Optional np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`, default = `None`
        states : Optional[np.ndarray] (M, ) of int
            State of each of the M binders being tracked, default = None
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        rad : Optional[float]
            Radius of the spherical ghost particle.
        """
        super(GhostBead, self).__init__(
            id_, r, t3, t2, states, binder_names
        )
        self.rad = rad
        self.vol = (4/3) * np.pi * rad ** 3
        self.kwargs = kwargs

    def test_collision(self):
        """Collisions are neglected for a ghost bead.
        """
        return False

    def print_properties(self):
        """Print properties of the ghost bead.
        """
        print("Ghost Bead ID: ", self.id)
        print("Central Position: ")
        print(self.r)
        print("t3 Orientation: ")
        print(self.t3)
        print("t2 Orientation: ")
        print(self.t2)


def get_verticies_rot_mat(current_vec, target_vec, fulcrum):
    """Rotate verticies defined relative to `current_vec`.

    Parameters
    ----------
    current_vec : array_like (3,) of double
        Current vector relative to which verticies are defined
    target_vec : array_like (3,) of double
        Target vector relative to which verticies should be defined
    fulcrum : array_like (3,) of double
        Point about which rotation will take place

    Returns
    -------
    array_like (4, 4) of double
        Homogeneous rotation matrix with which to rotate verticies
    """
    axis = np.cross(current_vec, target_vec)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(
        np.dot(current_vec, target_vec) /
        (np.linalg.norm(current_vec) * np.linalg.norm(target_vec))
    )
    return la.return_arbitrary_axis_rotation(axis, fulcrum, angle)


class Prism(Bead):
    """Class representation of prism-shaped monomer, enabling sterics.

    Notes
    -----
    The `Prism` objects allow for careful evaluation of collisions between
    nucleosome beads.

    This class is not yet tested.

    See documentation for `Bead` class for additional attribute definitions.

    Attributes
    ----------
    vertices : array_like (M, 3)
        Vertices representing a mesh of the nucleosome bead. The verticies
        are defined around the origin, with orientations such that t3
        coincides with the positive x axis and t2 coincides with the
        positive z axis.
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        vertices: np.ndarray, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize prism object.

        Parameters
        ----------
        id_ : int
            Identifier for the nucleosome
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : np.ndarray (3,)
            Tangent vector defining orientation of nucleosome
        t2 : np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`
        vertices : np.ndarray (M, 3)
            Vertices representing a mesh of the nucleosome bead. The verticies
            are defined around the origin, with orientations such that t3
            coincides with the positive x-axis and t2 coincides with the
            positive z-axis.
        states : Optional[np.ndarray] (M, ) of int
            State of each of the M epigenetic binders being tracked
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        """
        super(Prism, self).__init__(
            id_, r, t3, t2, states, binder_names
        )
        self.vertices = vertices

    @classmethod
    def construct_nucleosome(
        cls, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        num_sides: int, width: float, height: float,
        states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Construct nucleosome as a prism w/ specified position & orientation.

        Parameters
        ----------
        id_ : int
            Identifier for the nucleosome
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : np.ndarray (3,)
            Tangent vector defining orientation of nucleosome
        t2 : np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`
        num_sides : int
            Number of sides on the face of the prism used to represent the
            nucleosome's geometry; this determines the locations of verticies
            of the `Prism`
        width : float
            Determines the shape of the prism defining the location of the
            nucleosome's verticies. The `width` gives the diameter of the
            circle circumscribing the base of the prism in the simulation's
            distance units.
        height : float
            Determines the shape of the prism defining the location of the
            nucleosome's verticies. The `height` gives the height of the prism
            in the simulation's distance units.
        states : Optional[np.ndarray] (M,) of int
            State of each of the M epigenetic binders being tracked
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None

        Returns
        -------
        Prism
            Instance of a prism matching included specifications
        """
        verticies = la.get_prism_verticies(num_sides, width, height)
        return cls(
            id_, r, t3=t3, t2=t2, vertices=verticies, states=states,
            binder_names=binder_names
        )

    def test_collision(self, vertices: np.ndarray, max_iters: int) -> bool:
        """Test collision with the current nucleosome.

        Parameters
        ----------
        vertices : np.ndarray (M, 3)
            Vertices representing a mesh of the nucleosome bead
        max_iters : int
            Maximum iterations of the GJK algorithm to evaluate when testing
            for collision

        Returns
        -------
        bool
            Flag for collision with the nucleosome (True = collision, False =
            no collision)
        """
        return gjk_collision(self.transform_vertices(), vertices, max_iters)

    def print_properties(self):
        """Print properties of the current nucleosome.
        """
        print("Nucleosome ID: ", self.id)
        print("Central Position: ")
        print(self.r)
        print("t3 Orientation: ")
        print(self.t3)
        print("t2 Orientation: ")
        print(self.t2)

    def transform_vertices(self) -> np.ndarray:
        """Transform the verticies of the nucleosome based on `r`, `t2`, & `t3`.

        Notes
        -----
        Begin by translating the position of the verticies to match the
        position of the nucleosome in space.

        Then rotate the nucleosome so that the x-axis on which the verticies
        are defined aligns with the t3 tangent of the nucleosome.

        Finally, rotate the nucleosome so that the z-axis on which the vertices
        are defined aligns with the t2 tangent of the nucleosome.

        Returns
        -------
        np.ndarray (M, 3)
            Transformed vertices representing a mesh of the nucleosome bead
            positioned and oriented in space.
        """
        num_verticies = len(self.vertices)
        verticies = np.ones((num_verticies, 4))
        verticies[:, 0:3] = self.vertices
        verticies = verticies.T

        translate_mat = np.identity(4)
        for i in range(3):
            translate_mat[i, 3] = self.r[i]

        verticies = translate_mat @ verticies

        x_axis = np.array([1, 0, 0])
        if not np.allclose(x_axis, self.t3):
            rot_mat = get_verticies_rot_mat(x_axis, self.t3, self.r)
            verticies = rot_mat @ verticies

        z_axis = np.array([0, 0, 1])
        if not np.allclose(z_axis, self.t2):
            rot_mat = get_verticies_rot_mat(z_axis, self.t2, self.r)
            verticies = rot_mat @ verticies

        return verticies.T[:, 0:3]


class Nucleosome(Bead):
    """Class representation of a nucleosome bead.

    Notes
    -----
    See documentation for `Bead` class for additional attribute definitions.

    Attributes
    ----------
    rad : double
        Radius of spherical excluded volume around nucleosome
    vol : double
        Volume of spherical excluded volume around nucleosome
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        rad: Optional[float] = 5, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize nucleosome object.

        Parameters
        ----------
        id_ : int
            Identifier for the nucleosome
        r : array_like (3,) of double
            Coordinates of the nucleosome in form (x, y, z)
        t3 : array_like (3,) of double
            Tangent vector defining orientation of nucleosome
        t2 : array_like (3,) of double
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`
        rad : Optional[float]
            Radius of spherical excluded volume around nucleosome in simulation
            units of distance; default is 5
        states : Optional array_like (M,) of int
            State of each of the M epigenetic binders being tracked; default is
            None
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        """
        super(Nucleosome, self).__init__(id_, r, t3, t2, states, binder_names)
        self.rad = rad
        self.vol = (4/3) * np.pi * rad ** 3

    def test_collision(self, point: np.ndarray) -> bool:
        """Test collision with the nucleosome bead.

        Parameters
        ----------
        point : array_like (3,) of double
            Point at which to test for collision with the nucleosomes

        Returns
        -------
        bool
            Flag for collision with the nucleosome (True = collision, False =
            no collision)
        """
        if self.rad is not None:
            return np.linalg.norm(point - self.r) <= self.rad
        raise ValueError("Nucleosome radius is not specified.")

    def print_properties(self):
        """Print properties of the current nucleosome.
        """
        print("Nucleosome ID: ", self.id)
        print("Radius: ", self.rad)
        print("Position: ")
        print(self.r)
        print("t3 Orientation: ")
        print(self.t3)
        print("t2 Orientation: ")
        print(self.t2)


class DetailedNucleosome(Nucleosome):
    """A nucleosome with fixed entry/exit positions and orientations.

    Notes
    -----
    The entry and exit positions and orientations are defined relative to the
    position and t3/t2 vectors of the nucleosome. We assume at this stage that
    the relative entry and exit positions and orientations are fixed for all
    nucleosomes.

    The nucleosome is defined such that the orientation of the entering DNA is
    fixed by the t3 orientation. The orientation of the exiting DNA is dictated
    by the amount of DNA wrapping around the nucleosome. This class includes
    methods that return the positions and orientations for entering and
    exiting DNA strands based on the current position and orientation of the
    nucleosome.
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        bp_wrap: int, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize detailed nucleosome object.

        Notes
        -----
        Entering DNA must perfectly align with the tangent of the nucleosome.

        Parameters
        ----------
        see documentation for `Nucleosome` class

        bp_wrap : int
            Number of base pairs wrapped around nucleosome
        """
        rad = consts_dict["R"]
        super(DetailedNucleosome, self).__init__(
            id_, r, t3=t3, t2=t2, rad=rad,
            states=states, binder_names=binder_names
        )
        self.bp_wrap = bp_wrap
        self.length_wrap = (bp_wrap-1) * LENGTH_BP

        # Specify local reference frame
        self.t3_local = np.array([
            0,
            2 * np.pi * rad / consts_dict["Lt"],
            consts_dict["h"] / consts_dict["Lt"]
        ])
        self.t2_local = np.array([
            0,
            -consts_dict["h"] / consts_dict["Lt"],
            2 * np.pi * rad / consts_dict["Lt"]
        ])
        self.r_enter_local = np.array([
            rad,
            0,
            -(consts_dict["h"] * consts_dict["s"] / consts_dict["Lt"]) / 2
        ])
        self.r_exit_local = get_r(self.length_wrap, consts_dict)

        # Get normalized positions
        self.r_enter_local_norm = np.linalg.norm(self.r_enter_local)
        self.r_exit_local_norm = np.linalg.norm(self.r_exit_local)
        self.r_enter_local_unit = self.r_enter_local / self.r_enter_local_norm
        self.r_exit_local_unit = self.r_exit_local / self.r_exit_local_norm

        # Update configuration
        self.update_configuration(self.r, self.t3, self.t2)

    def get_entry_exit_positions(self):
        """Get entry and exit positions of DNA in the global reference frame.
        """
        self.r_enter = \
            np.dot(self.R_local_to_global, self.r_enter_local_unit) \
            * self.r_enter_local_norm + np.asarray(self.r)
        self.r_exit = \
            np.dot(self.R_local_to_global, self.r_exit_local_unit) \
            * self.r_exit_local_norm + np.asarray(self.r)

    def get_entry_exit_orientations(self):
        """Get entry and exit orientations of DNA in global reference frame.
        """
        t3_exit, t2_exit, t1_exit = get_exiting_orientations(
            self.length_wrap, self.t3, self.t2, self.t1, consts_dict)
        return t3_exit, t2_exit, t1_exit

    def update_configuration(self, r, t3, t2):
        """Update the position and orientations of the nucleosome.

        Parameters
        ----------
        r : np.ndarray (3,) of float
            Position of nucleosome in global reference frame
        t3, t2 : np.ndarray (3,) of float
            Orthogonal unit vectors defining the orientation of the nucleosome
            in the global reference frame.

        Returns
        -------
        r_enter : np.ndarray (3,) of float
            Entry position of DNA defined in global reference frame
        r_exit : np.ndarray (3,) of float
            Exit position of DNA defined in global reference frame
        t3 : np.ndarray (3,) of float
            t3 tangent of entering DNA in global reference frame
        t3_exit : np.ndarray (3,) of float
            t3 tangent of exiting DNA in global reference frame
        t2 : np.ndarray (3,) of float
            t2 tangent of entering DNA in global reference frame
        t2_exit : np.ndarray (3,) of float
            t2 tangent of exiting DNA in global reference frame
        """
        # Update user-specified geometry
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.t1 = np.cross(self.t2, self.t3)
        # Compute the rotation matrix that rotates local to global coordinates
        self.R_local_to_global = la.get_rotation_matrix(
            self.t3_local, self.t2_local, self.t3, self.t2
        )
        # Compute configuration in a global coordinate system
        self.get_entry_exit_positions()
        t3_exit, t2_exit, t1_exit = self.get_entry_exit_orientations()
        return self.r_enter, self.r_exit, self.t3, t3_exit, self.t2, t2_exit
