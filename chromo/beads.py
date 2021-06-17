"""Beads represent monomeric units forming the polymer.
"""

# Built-in Modules
from abc import ABC, abstractmethod
from typing import (Sequence, Optional)

# External Modules
import numpy as np

# Custom Modules
from .util import linalg as la
from .util.gjk import gjk_collision


class Bead(ABC):
    """Abstract class representation of a bead of a polymer.
    """

    @abstractmethod
    def test_collision(self):
        """Test collisions with another bead or a confinement.
        """
        pass

    @abstractmethod
    def print_properties(self):
        """Print properties of the bead.
        """
        pass


class DetailedNucleosome(Bead):
    """Class representation of a detailed nucleosome bead.

    The `DetailedNucleosome` objects allow for careful evaluation of collisions
    between nucleosome beads.
    """

    def __init__(
        self,
        id: int,
        r: np.ndarray,
        t3: np.ndarray,
        t2: np.ndarray,
        vertices: np.ndarray,
        states: Optional[np.ndarray] = None,
        mark_names: Optional[Sequence[str]] = None
    ):
        """Initialize detailed nucleosome object.

        Parameters
        ----------
        id : int
            Identifier for the nucleosome
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3, t2 : np.ndarray (3,)
            Tangent vectors defining orientation of nucleosomes
        vertices: np.ndarray (M, 3)
            Vertices representing a mesh of the nucleosome bead. The verticies
            are defined around the origin, with orientations such that t3
            coincides with the positive x axis and t2 coincides with the
            positive z axis.
        states : (M, ) array_like of int
            State of each of the M epigenetic marks being tracked
        mark_names : (M, ) str or Sequence[str]
            The name of each chemical modification tracked in `states`, for
            each of tracking which mark is which.
        """
        self.id = id
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.vertices = vertices
        self.states = states
        self.mark_names = mark_names

    @classmethod
    def construct_nucleosome(
        cls,
        id: int,
        r: np.ndarray,
        t3: np.ndarray,
        t2: np.ndarray,
        num_sides: int,
        width: float,
        height: float,
        states: Optional[np.ndarray] = None,
        mark_names: Optional[Sequence[str]] = None
    ):
        """Construct nucleosome as a prism w/ specified position & orientation.

        Parameters
        ----------
        id : int
            Identifier for the nucleosome
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3, t2 : np.ndarray (3,)
            Tangent vectors defining orientation of nucleosomes
        num_sides : int
            Number of sides on the face of the prism used to represent the
            nucleosome's geometry; this determines the locations of verticies
            of the `DetailedNucleosome`
        width, height : float
            Determines the shape of the prism defining the location of the
            nucleosome's verticies. The `width` gives the diameter of the
            circle circumscribing the base of the prism. The `height` gives
            the height of the prism.
        states : (M, ) array_like of int
            State of each of the M epigenetic marks being tracked
        mark_names : (M, ) str or Sequence[str]
            The name of each chemical modification tracked in `states`, for
            each of tracking which mark is which.
        """
        verticies = la.get_prism_verticies(num_sides, width, height)
        return cls(id, r, t3, t2, verticies, states, mark_names)

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
        """Transform the verticies of the nucleosome based on r, t2, and t3.

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
        num_verticies = len(self.verticies)
        verticies = np.ones((num_verticies, 4))
        verticies[:, 0:3] = self.vertices
        verticies = verticies.T

        translate_mat = la.generate_translation_mat(
            self.r[0], self.r[1], self.r[2]
        )
        verticies = translate_mat @ verticies

        x_axis = np.array([1, 0, 0])
        if not np.allclose(x_axis, self.t3):
            rot_mat = self.get_verticies_rot_mat(x_axis, self.t3, self.r)
            verticies = rot_mat @ verticies

        z_axis = np.array([0, 0, 1])
        if not np.allclose(z_axis, self.t2):
            rot_mat = self.get_verticies_rot_mat(z_axis, self.t2, self.r)
            verticies = rot_mat @ verticies

        return verticies.T[:, 0:3]

    def get_verticies_rot_mat(self, current_vec, target_vec, fulcrum):
        """Rotate verticies defined relative to `current_vec`.

        Parameters
        ----------
        current_vec : np.ndarray (3, )
            Current vector relative to which verticies are defined
        target_vec : np.ndarray (3, )
            Target vector relative to which verticies should be defined
        fulcrum : np.ndarray (3, )
            Point about which rotation will take place

        Returns
        -------
        np.ndarray (4, 4)
            Homogeneous rotation matrix with which to rotate verticies
        """
        axis = np.cross(current_vec, target_vec)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(
            np.dot(current_vec, target_vec) /
            (np.linalg.norm(current_vec) * np.linalg.norm(target_vec))
        )
        return la.arbitrary_axis_rotation(axis, fulcrum, angle)


class Nucleosome(Bead):
    """Class representation of a nucleosome bead.
    """

    def __init__(
        self,
        id: int,
        r: np.ndarray,
        t3: np.ndarray,
        t2: np.ndarray,
        bead_length: float,
        rad: Optional[float] = None,
        states: Optional[np.ndarray] = None,
        mark_names: Optional[Sequence[str]] = None
    ):
        """Initialize nucleosome object.

        Parameters
        ----------
        id : int
            Identifier for the nucleosome
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3, t2 : np.ndarray (3,)
            Tangent vectors defining orientation of nucleosomes
        bead_length : float
            Spacing between the nucleosome and its neighbor
        rad : Optional[float]
            Radius of spherical excluded volume around nucleosome
        states : (M, ) array_like of int
            State of each of the M epigenetic marks being tracked
        mark_names : (M, ) str or Sequence[str]
            The name of each chemical modification tracked in `states`, for
            each of tracking which mark is which.
        """
        self.id = id
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.bead_length = bead_length
        self.rad = rad
        self.states = states
        self.mark_names = mark_names

    def test_collision(self, point: np.ndarray) -> bool:
        """Test collision with the nucleosome bead.

        Parameters
        ----------
        point : np.ndarray
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
