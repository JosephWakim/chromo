"""Modifications to polymer's reader protein binding state that affect energy.

Notes
-----
Each simulation will typically involve a polymer bound by some `Binder`.

For common, physically characterized binding components, such as those bound to
known epigenetic modifications or well-characterized modifications on other real
polymers, the best known parameters for that binder should be documented here as
instances of the appropriate `Binder` subclass.
"""
import inspect
import sys

import pandas as pd
import numpy as np


cdef class Binder:
    """Class representation of an arbitrary components binding to a polymer.

    Notes
    -----
    For this code to work, all binders need a string name.

    Attributes
    ----------
    name : str
        Name of the binding component
    sites_per_bead : int
        Number of sites to which the binding component can bind on each
        monomeric unit of the polymer
    """

    def __init__(self, name: str, sites_per_bead: int) -> None:
        """Initialize the Binder object.

        Parameters
        ----------
        name : str
            Name of the binding component
        sites_per_bead : int
            Number of binding components that can be bound to a single polymer
            bead
        """
        self.name = name
        self.sites_per_bead = sites_per_bead
        self.binding_seq = np.zeros((sites_per_bead,), dtype=int)


cdef class ReaderProtein(Binder):
    """Information about the chemical properties of a reader protein.
    """

    def __init__(
        self,
        name: str,
        sites_per_bead: int,
        bind_energy_mod: float,
        bind_energy_no_mod: float,
        interaction_energy: float,
        chemical_potential: float,
        interaction_radius: float,
        cross_talk_interaction_energy: dict = {}
    ) -> None:
        """Initialize the reader protein object with physical properties.

        Parameters
        ----------
        name : str
            Name of the reader protein binding the polymer
        sites_per_bead : int
            Number of the reader protein that can bind a single polymer bead
        bind_energy_mod : float
            Binding energy of the reader protein to a bead with the associated
            chemical modification
        bind_energy_no_mod : float
            Binding energy of the reader protein to a bead without the
            associated chemical modification
        interaction_energy : float
            Interaction energy contributed by each pair of the reader protein
            within a specified interaction volume of one-another -- e.g., can
            represent oligomerization between reader proteins
        chemical_potential : float
            Chemical potential of the unbound reader proteins; characterizes
            the tendency for the reader protein to bind the polymer and is
            dicated by the concentration of unbound reader proteins
        interaction_radius : float
            Cutoff distance between pairs of reader proteins for which the
            interaction energy will be acquired
        cross_talk_interaction_energy : Dict[str, float]
            Cross talk interaction energy between the reader protein and other
            reader proteins in the system
        """
        super().__init__(name, sites_per_bead)
        self.bind_energy_mod = bind_energy_mod
        self.bind_energy_no_mod = bind_energy_no_mod
        self.interaction_energy = interaction_energy
        self.chemical_potential = chemical_potential
        self.interaction_radius = interaction_radius
        self.interaction_volume = (4.0/3.0) * np.pi * interaction_radius ** 3
        self.field_energy_prefactor = 0.0
        self.interaction_energy_intranucleosome = 0.0
        self.cross_talk_interaction_energy = cross_talk_interaction_energy
        self.cross_talk_field_energy_prefactor = {}
    
    def dict(self):
        """Represent a class instance as a dictionary (exclude derived values).

        Returns
        -------
        dict
            Dictionary with key attributes representing the reader protein
        """
        return {
            "name": self.name,
            "sites_per_bead": self.sites_per_bead,
            "bind_energy_mod": self.bind_energy_mod,
            "bind_energy_no_mod": self.bind_energy_no_mod,
            "interaction_energy": self.interaction_energy,
            "chemical_potential": self.chemical_potential,
            "interaction_radius": self.interaction_radius,
            "interaction_volume": self.interaction_volume,
            "field_energy_prefactor": self.field_energy_prefactor,
            "interaction_energy_intranucleosome":\
                self.interaction_energy_intranucleosome,
            "cross_talk_interaction_energy": self.cross_talk_interaction_energy,
            "cross_talk_field_energy_prefactor":\
                self.cross_talk_field_energy_prefactor
        }


null_reader = ReaderProtein(
    'null_reader', sites_per_bead=0, bind_energy_mod=0, bind_energy_no_mod=0,
    interaction_energy=0, chemical_potential=0, interaction_radius=0
)
"""Placeholder reader protein.

Notes
-----
Right now, the simulator requires at least one reader protein to be defined;
this serves as a placeholder.
"""


hp1 = ReaderProtein(
    'HP1', sites_per_bead=2, bind_energy_mod=-0.01, bind_energy_no_mod=1.52,
    interaction_energy=-4, chemical_potential=-1, interaction_radius=3,
    cross_talk_interaction_energy={'PRC1': 0}
)
"""Heterochromatin Protein 1, binds H3K9me marks.

Notes
-----
For now, we just use filler values for the energies. In the future, this
documentation string should go through the process of explaining exactly how we
arrive at the values that we actually use.

The filler values for interaction energy and chemical potential are both 1.

We define a cross-talk interaction energy to capture the relationship between
HP1 and PRC1; the default value of zero suggests no cross-talk, indicating that
HP1 and PRC1 occur completely independent of one-another.

To avoid double-counting the cross-talk between HP1 and PRC1, we do not include
a cross-talk interaction energy for PRC1.
"""

prc1 = ReaderProtein(
    'PRC1', sites_per_bead=2, bind_energy_mod=-0.01, bind_energy_no_mod=1.52,
    interaction_energy=-4, chemical_potential=-1, interaction_radius=3,
    cross_talk_interaction_energy={'HP1': 0}
)
"""Protein Regulator of Cytokinesis 1, binds H3K27me marks.

Notes
-----
For now, we just use filler values for the energies. In the future, this
documentation string should go through the process of explaining exactly how we
arrive at the values that we actually use.
The filler values for interaction energy and chemical potential are both 1.
"""


def get_by_name(name):
    """Look up saved reader protein by name.

    Parameters
    ----------
    name : str
        Name of the pre-defined reader protein to retrieve

    Returns
    -------
    Binder
        Object representing the reader protein that was queried.
    """
    all_binders = [
        obj for name_, obj in inspect.getmembers(sys.modules[__name__])
        if isinstance(obj, Binder)
    ]
    matching_binders = [binder for binder in all_binders if binder.name == name]
    if not matching_binders:
        raise ValueError(f"No binders found in {__name__} with name: {name}")
    if len(matching_binders) > 1:
        raise ValueError(f"More than one binder has the name requested: {name}")
    return matching_binders[0]


def make_binder_collection(binders):
    """Construct summary DataFrame from sequence of binders.

    Parameters
    ----------
    binders : Binder or str or Sequence[Binder] or Sequence[str]
        The binders to be summarized by the DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns are the properties of each Binders.
    """
    df = pd.DataFrame(
        columns=[
            'name', 'sites_per_bead', 'bind_energy',
            'interaction_energy', 'chemical_potential'
        ]
    )
    if binders is None:
        return None
    input_type = type(binders)
    if input_type is str or issubclass(input_type, Binder):
        binders = [binders]  # allow the "one binder" case
    for binder in binders:
        if type(binder) is str:
            binder = get_by_name(binder)
        df = df.append(binder.dict(), ignore_index=True)
    return df
