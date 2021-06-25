"""Monte Carlo simulations of a discrete wormlike chain.

This module runs Monte Carlo simulations for the reconfiguration of multiple
discrete wormlike chains when placed in a user-specified field and labeled by
multiple epigenetic marks.
"""

# Built-in Modules
from pathlib import Path
# import warnings
from typing import List, Optional, Callable, TypeVar

# External Modules
import numpy as np

# Custom Modules
from .mc_sim import mc_sim
# from .moves import move_list
from .mc_controller import all_moves, Controller, SimpleControl
from ..util.reproducibility import make_reproducible
from ..polymers import Polymer, Chromatin
from ..marks import Epigenmark
from ..marks import get_by_name, make_mark_collection
from ..fields import UniformDensityField

F = TypeVar("F")    # Represents an arbitrary field

# Custom Data Types
STEPS = int      # Number of steps per MC save point
SAVES = int      # Number of save points
SEED = int       # Random seed
DIR = str        # Directory in which to save outputs


@make_reproducible
def simple_mc(
    num_polymers: int, num_beads: int, bead_length: float,
    num_marks: int, num_save_mc: int, num_saves: int,
    x_width: float, nx: int, y_width: float, ny: int,
    z_width: float, nz: int, random_seed: Optional[SEED] = 0,
    output_dir: Optional[DIR] = '.'
) -> Callable[
    [List[Chromatin], List[Epigenmark], F, STEPS, SAVES, SEED, DIR], None
]:
    """Single line implementation of basic Monte Carlo simulation.

    Initialize straight-line polymers with HP1 marks, and simulate in a uniform
    density field.

    Parameters
    ----------
    num_polymers : int
        Number of polymers in the Monte Carlo simulation
    num_beads : int
        Number of beads in each polymer of the Monte Carlo simulation
    bead_length : float
        Length associated with a single bead in the polymer (bead + linker)
    num_marks : int
        Number of epigenetic marks in the simulation
    num_save_mc : int
        Number of Monte Carlo steps to take between configuration save points
    num_saves : int
        Number of save points to make in Monte Carlo simulation
    x_width, y_width, z_width : float
        x,y,z-direction bin widths when discretizing space
    nx, ny, nz : int
        Number of bins in the x,y,z-direction when discretizing space
    random_seed : Optional[int]
        Random seed to apply in simulation for reproducibility (default = 0)
    output_dir : Optional[str]
        Path to output directory in which polymer configurations will be saved
        (default = '.')

    Returns
    -------
    Callable[[List[Chromtin], List[Epigenmark], F, STEPS, SAVES, SEED,
    DIR], None]
        Monte Carlo simulation of a tssWLC in a field
    """
    polymers = [
        Chromatin.straight_line_in_x(
            f'Polymer-{i}', num_beads, bead_length,
            states=np.zeros((num_beads, num_marks)),
            mark_names=num_marks*['HP1']
        ) for i in range(num_polymers)
    ]
    marks = [get_by_name('HP1') for i in range(num_marks)]
    marks = make_mark_collection(marks)
    field = UniformDensityField(polymers, marks, x_width, nx, y_width, ny,
                                z_width, nz)
    return _polymer_in_field(polymers, marks, field, num_save_mc, num_saves,
                             random_seed=random_seed, output_dir=output_dir)


def _polymer_in_field(
    polymers: List[Polymer], marks: List[Epigenmark],
    field: F, num_save_mc: STEPS, num_saves: SAVES,
    mc_move_controllers: Optional[List[Controller]] = None,
    random_seed: Optional[SEED] = 0, output_dir: Optional[DIR] = '.'
):
    """
    Monte Carlo simulation of a tssWLC in a field.

    Identify the active Monte Carlo moves, and for each save point, perform a
    Monte Carlo simulation and log coordinates and move/bead amplitudes.

    Parameters
    ----------
    polymers : List[Polymer]
        The polymers to be simulated
    epigenmarks : List[Epigenmark]
        Output of `chromo.marks.make_mark_collection`. Summarizes the energetic
        properties of each chemical modification
    field : F
        The discretization of space in which to simulate the polymers
    num_save_mc : int
        Number of Monte Carlo steps to take between configuration save points
    num_saves : int
        Number of save points to make in Monte Carlo simulation
    mc_move_controllers : Optional[List[Controller]]
        Controllers for monte carlo moves desired; default of `None` activates
        `SimpleControl` for all MC moves
    random_seed : Optional[SEED]
        Random seed for replication of simulation (default = 0)
    output_dir : Optional[Path]
        Path to output directory in which polymer configurations will be saved
        (default = '.')
    """
    poly_len = np.min([polymer.r.shape[0] for polymer in polymers])
    bead_amp_bounds = {
        "crank_shaft": (20, poly_len),
        "slide": (4, poly_len),
        "end_pivot": (
            int(poly_len/2 - poly_len/20), int(poly_len/2 + poly_len/20)
        ),
        "tangent_rotation": (1, poly_len)
    }
    move_amp_bounds = {
        "crank_shaft": (0.05, np.pi),
        "slide": (0.05, 1),
        "end_pivot": (0.05, np.pi),
        "tangent_rotation": (0.05, np.pi)
    }
    if mc_move_controllers is None:
        mc_move_controllers = all_moves(
            log_dir=output_dir,
            polymers=polymers,
            bead_amp_bounds=bead_amp_bounds,
            move_amp_bounds=move_amp_bounds,
            controller=SimpleControl
        )
    for mc_count in range(num_saves):
        mc_sim(polymers, marks, num_save_mc, mc_move_controllers, field)
        for poly in polymers:
            poly.to_csv(
                output_dir / Path(f"{poly.name}-{mc_count}.csv")
            )
        for controller in mc_move_controllers:
            controller.move.acceptance_tracker.save_move_log(
                snapshot=mc_count
            )
        print("Save point " + str(mc_count) + " completed")
    for polymer in polymers:
        polymer.update_log_path(
            str(output_dir) + "/" + polymer.name + "_config_log.csv"
        )

    # warnings.warn("The random seed is currently ignored.", UserWarning)


polymer_in_field = make_reproducible(_polymer_in_field)
