"""Monte Carlo simulations of a discrete wormlike chain.

This module runs Monte Carlo simulations for the reconfiguration of multiple
discrete wormlike chains when placed in a user-specified field and labeled by
multiple epigenetic marks.
"""

from pathlib import Path
from typing import List, Optional, Callable, TypeVar, Dict, Tuple

import numpy as np

from .mc_sim import mc_sim
from .mc_controller import all_moves, Controller, SimpleControl
from .moves import Bounds
from ..util.reproducibility import make_reproducible
from ..util.poly_stat import (
    find_polymers_in_output_dir, get_latest_configuration,
    get_latest_simulation
)
from ..util.timer import decorator_timed_path
from ..polymers import Polymer, Chromatin
from ..marks import Epigenmark
from ..marks import get_by_name, make_mark_collection
from ..fields import UniformDensityField


F = TypeVar("F")    # Represents an arbitrary field
STEPS = int         # Number of steps per MC save point
SAVES = int         # Number of save points
SEED = int          # Random seed
DIR = str           # Directory in which to save outputs


def _polymer_in_field(
    polymers: List[Polymer],
    marks: List[Epigenmark],
    field: F,
    num_save_mc: STEPS,
    num_saves: SAVES,
    bead_amp_bounds: Dict[str, Tuple[int, int]],
    move_amp_bounds: Dict[str, Tuple[int, int]],
    mc_move_controllers: Optional[List[Controller]] = None,
    random_seed: Optional[SEED] = 0,
    output_dir: Optional[DIR] = '.',
    **kwargs
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
    bead_amp_bounds : Dict[str, Tuple[int, int]]
        Dictionary of bead selection bounds for each move type, where keys are
        the names of the move types and values are tuples in the form (lower
        bound, upper bound)
    move_amp_bounds : Dict[str, Tuple[int, int]]
        Dictionary of move amplitude bounds for each move type, where keys are
        the names of the move types and values are tuples in the form (lower
        bound, upper bound)
    mc_move_controllers : Optional[List[Controller]]
        Controllers for monte carlo moves desired; default of `None` activates
        `SimpleControl` for all MC moves
    random_seed : Optional[SEED]
        Random seed for replication of simulation (default = 0)
    output_dir : Optional[Path]
        Path to output directory in which polymer configurations will be saved
        (default = '.')
    """
    np.random.seed(random_seed)
    if mc_move_controllers is None:
        mc_move_controllers = all_moves(
            log_dir=output_dir,
            polymers=polymers,
            bead_amp_bounds=bead_amp_bounds.bounds,
            move_amp_bounds=move_amp_bounds.bounds,
            controller=SimpleControl
        )
    for mc_count in range(num_saves):
        decorator_timed_path(output_dir)(mc_sim)(
            polymers, marks, num_save_mc, mc_move_controllers, field
        )
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


polymer_in_field = make_reproducible(_polymer_in_field)


def continue_polymer_in_field_simulation(
    polymer_class: Polymer,
    marks: List[Epigenmark],
    fields: F,
    output_dir: str,
    num_save_mc: STEPS,
    num_saves: SAVES,
    mc_move_controllers: Optional[List[Controller]] = None,
    random_seed: Optional[SEED] = 0
):
    """Continue a simulation of a polymer in a field.

    Parameters
    ----------
    polymer_class : Polymer
        Class of polymers for which to continue the simulation
    marks : List[Epigenmark]
        List of epigenetic marks active on polymer
    field : F
        The discretization of space in which to simulate the polymers
    output_dir : str
        Output directory from which to continue the simulation
    num_save_mc : int
        Number of steps per snapshot in continued simulation
    num_saves : int
        Number of additional save points to collect
    mc_move_controllers : Optional[List[Controller]]
        Controllers for monte carlo moves desired; default of `None` activates
        `SimpleControl` for all MC moves
    random_seed : Optional[SEED]
        Random seed for replication of simulation (default = 0)
    """
    latest_output_subdir = get_latest_simulation(output_dir)
    latest_output_subdir_path = output_dir + "/" + latest_output_subdir
    polymer_names = find_polymers_in_output_dir(latest_output_subdir_path)
    latest_config_paths = [
        get_latest_configuration(
            polymer_prefix=polymer_name, dir_=latest_output_subdir_path
        ) for polymer_name in polymer_names
    ]
    latest_config_names = [
        "-".join(path.split("/")[-1].split(".")[0].split("-")[0:2])
        for path in latest_config_paths
    ]
    polymers = [
        polymer_class.from_file(
            latest_config_paths[i], latest_config_names[i]
        ) for i in range(len(latest_config_paths))
    ]
    bead_amp_bounds, move_amp_bounds = get_amplitude_bounds(polymers)
    args = [
        polymers, marks, fields, num_save_mc, num_saves, bead_amp_bounds,
        move_amp_bounds
    ]
    if mc_move_controllers is not None:
        args.append(mc_move_controllers)
    polymer_in_field(
        *args, random_seed=random_seed, output_dir=output_dir,
        continue_from=latest_output_subdir
    )


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
    field = UniformDensityField(
        polymers, marks, x_width, nx, y_width, ny, z_width, nz
    )
    return _polymer_in_field(
        polymers, marks, field, num_save_mc, num_saves,
        random_seed=random_seed, output_dir=output_dir
    )


def get_amplitude_bounds(
    polymers: List[Polymer]
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[float, float]]]:
    """Get lower and upper bound for bead selection and move amplitudes.

    Parameters
    ----------
    polymers : List[Polymer]
        List of polymers involved in simulation

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Dictionary of bead selection bounds for each move type, where keys are
        the names of the move types and values are tuples in the form (lower
        bound, upper bound)
    Dict[str, Tuple[int, int]]
        Dictionary of move amplitude bounds for each move type, where keys are
        the names of the move types and values are tuples in the form (lower
        bound, upper bound)
    """
    poly_len = np.min([polymer.r.shape[0] for polymer in polymers])
    bead_amp_bounds = Bounds("bead_amp_bounds", {
        "crank_shaft": (20, poly_len),
        "slide": (4, poly_len),
        "end_pivot": (int(0.45 * poly_len), int(0.55 * poly_len)),
        "tangent_rotation": (1, poly_len)
    })
    move_amp_bounds = Bounds("move_amp_bounds", {
        "crank_shaft": (0.05, np.pi),
        "slide": (0.05, 1),
        "end_pivot": (0.05, np.pi),
        "tangent_rotation": (0.05, np.pi)
    })
    return bead_amp_bounds, move_amp_bounds
