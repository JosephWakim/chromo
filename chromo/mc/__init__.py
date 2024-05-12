"""Monte Carlo simulations of a discrete wormlike chain.

This module runs Monte Carlo simulations for the reconfiguration of multiple
discrete wormlike chains when placed in a user-specified field and labeled by
multiple epigenetic marks.
"""

from pathlib import Path
from typing import List, Optional, Callable, TypeVar, Dict, Tuple
from time import process_time

import numpy as np

from chromo.mc.mc_sim import mc_sim
from chromo.mc.mc_controller import all_moves, Controller, SimpleControl
from chromo.mc.moves import Bounds
from chromo.util.reproducibility import make_reproducible
from chromo.util.poly_stat import (
    find_polymers_in_output_dir, get_latest_configuration,
    get_latest_simulation
)
from chromo.util.timer import decorator_timed_path
from chromo.polymers import PolymerBase, Chromatin
from chromo.binders import ReaderProtein
from chromo.binders import get_by_name, make_binder_collection
from chromo.fields import UniformDensityField, FieldBase


F = TypeVar("F")    # Represents an arbitrary field
STEPS = int         # Number of steps per MC save point
SAVES = int         # Number of save points
SEED = int          # Random seed
DIR = str           # Directory in which to save outputs


def _polymer_in_field(
    polymers: List[PolymerBase],
    binders: List[ReaderProtein],
    field: FieldBase,
    num_save_mc: STEPS,
    num_saves: SAVES,
    bead_amp_bounds: Dict[str, Tuple[int, int]],
    move_amp_bounds: Dict[str, Tuple[int, int]],
    mc_move_controllers: Optional[List[Controller]] = None,
    random_seed: Optional[int] = 0,
    mu_schedule: Optional[Callable[[float], float]] = None,
    output_dir: Optional[DIR] = '.',
    path_to_run_script: Optional[str] = None,
    path_to_chem_mods: Optional[List[str]] = None,
    run_command: Optional[str] = None,
    **kwargs
):
    """
    Monte Carlo simulation of a tssWLC in a field.

    Identify the active Monte Carlo moves, and for each save point, perform a
    Monte Carlo simulation and log coordinates and move/bead amplitudes.

    Parameters
    ----------
    polymers : List[PolymerBase]
        The polymers to be simulated
    binders : List[ReaderProtein]
        Output of `chromo.binders.make_binder_collection`. Summarizes the
        energetic properties of each chemical modification
    field : FieldBase
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
    random_seed : Optional[int]
        Random seed for replication of simulation (default = 0)
    mu_schedule : Optional[Callable[[int, int], float]]
        Function returning factor adjustment to chemical potential defining
        simulated annealing; the function takes two arguments: the first is the
        current snapshot of the simulation, and the second is the total number
        number of snapshots run during the simulation (default = None;
        indicating no simulated annealing will be applied)
    output_dir : Optional[Path]
        Path to output directory in which polymer configurations will be saved
        (default = '.')
    path_to_run_script : Optional[str]
        Path to the python file called to run the simulation; if this argument
        is provided, the python file will be copied to the output directory
        (default = None).
    path_to_chem_mods : Optional[List[str]]
        Paths to the text files containing chemical modification patterns
        associated with the simulation; if this argument is provided, the
        chemical modification patterns will be copied to the output directory
        (default = None)
    run_command : Optional[str]
        Command called in the console to run the simulation; if this argument is
        provided, the run command will be copied to the output directory
        (default = None)
    """
    np.random.seed(random_seed)
    if mc_move_controllers is None:
        mc_move_controllers = all_moves(
            log_dir=output_dir,
            bead_amp_bounds=bead_amp_bounds.bounds,
            move_amp_bounds=move_amp_bounds.bounds,
            controller=SimpleControl
        )

    if path_to_run_script is not None:
        print(f"Running simulation from file: \n    {path_to_run_script}\n")
    if run_command is not None:
        print(f"Running simulation using command: \n    {run_command}\n")
    if path_to_chem_mods is not None:
        print("Loading chemical modification patterns from files: ")
        for path in path_to_chem_mods:
            print(path)
        print()

    t1_start = process_time()
    for mc_count in range(num_saves):

        # Simulated annealing
        if mu_schedule is not None:
            mu_adjust_factor = mu_schedule.function(mc_count, num_saves)
        else:
            mu_adjust_factor = 1
        inner_seed = np.random.randint(0, 1E5)
        decorator_timed_path(output_dir)(mc_sim)(
            polymers, binders, num_save_mc, mc_move_controllers, field,
            mu_adjust_factor, inner_seed
        )
        for poly in polymers:
            poly.to_csv(
                str(output_dir / Path(f"{poly.name}-{mc_count}.csv"))
            )
        for controller in mc_move_controllers:
            controller.move.acceptance_tracker.create_log_file(mc_count)
            controller.move.acceptance_tracker.save_move_log(
                snapshot=mc_count
            )
        print("Save point " + str(mc_count) + " completed")

    for polymer in polymers:
        polymer.update_log_path(
            str(output_dir) + "/" + polymer.name + "_config_log.csv"
        )

    print(
        "Simulation Runtime (in seconds): ", round(
             process_time()-t1_start, 2
        )
    )
    return polymers


polymer_in_field = make_reproducible(_polymer_in_field)


def continue_polymer_in_field_simulation(
    polymer_class,
    binders: List[ReaderProtein],
    field: FieldBase,
    output_dir: str,
    num_save_mc: STEPS,
    num_saves: SAVES,
    mc_move_controllers: Optional[List[Controller]] = None,
    random_seed: Optional[SEED] = 0
):
    """Continue a simulation of a polymer in a field.

    Parameters
    ----------
    polymer_class : PolymerBase
        Class of polymers for which to continue the simulation
    binders : List[ReaderProteins]
        List of reader proteins active on polymer
    field : FieldBase
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
            polymer_prefix=polymer_name, directory=latest_output_subdir_path
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
    field.polymers = polymers
    bead_amp_bounds, move_amp_bounds = get_amplitude_bounds(polymers)
    args = [
        polymers, binders, field, num_save_mc, num_saves, bead_amp_bounds,
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
    num_binders: int, num_save_mc: int, num_saves: int,
    x_width: float, nx: int, y_width: float, ny: int,
    z_width: float, nz: int, random_seed: Optional[int] = 0,
    output_dir: Optional[DIR] = '.'
) -> Callable[
    [List[Chromatin], List[ReaderProtein], F, STEPS, SAVES, int, DIR], None
]:
    """Single line implementation of basic Monte Carlo simulation.

    Initialize straight-line polymers with HP1 reader proteins, and simulate
    in a uniform density field.

    Parameters
    ----------
    num_polymers : int
        Number of polymers in the Monte Carlo simulation
    num_beads : int
        Number of beads in each polymer of the Monte Carlo simulation
    bead_length : float
        Length associated with a single bead in the polymer (bead + linker)
    num_binders : int
        Number of reader proteins in the simulation
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
    Callable[[List[Chromtin], List[ReaderProteins], FieldBase, STEPS, SAVES,
    int, DIR], None]
        Monte Carlo simulation of a tssWLC in a field
    """
    polymers = [
        Chromatin.straight_line_in_x(
            f'Polymer-{i}', np.ones(num_beads) * bead_length,
            states=np.zeros((num_beads, num_binders)),
            binder_names=num_binders*['HP1']
        ) for i in range(num_polymers)
    ]
    binders = [get_by_name('HP1') for _ in range(num_binders)]
    binders = make_binder_collection(binders)
    field = UniformDensityField(
        polymers, binders, x_width, nx, y_width, ny, z_width, nz
    )
    bead_amp_bounds, move_amp_bounds = get_amplitude_bounds(polymers)
    return _polymer_in_field(
        polymers, binders, field, num_save_mc, num_saves,
        random_seed=random_seed, output_dir=output_dir,
        bead_amp_bounds=bead_amp_bounds, move_amp_bounds=move_amp_bounds
    )


def get_amplitude_bounds(
    polymers: List[PolymerBase]
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[float, float]]]:
    """Get lower and upper bound for bead selection and move amplitudes.

    Parameters
    ----------
    polymers : List[PolymerBase]
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
    min_spacing = np.min([np.min(polymer.bead_length) for polymer in polymers])
    bead_amp_bounds = Bounds("bead_amp_bounds", {
        "crank_shaft": (min(30, poly_len), min(150, poly_len)),
        "slide": (min(10, poly_len), min(150, poly_len)),
        "end_pivot": (min(50, poly_len/4), min(150, int(poly_len/2))),
        "tangent_rotation": (1, poly_len),
        "change_binding_state": (1, 1)
    })
    move_amp_bounds = Bounds("move_amp_bounds", {
        "crank_shaft": (0.1 * np.pi, 0.25 * np.pi),
        "slide": (0.2 * min_spacing, 0.3 * min_spacing),
        "end_pivot": (0.2 * np.pi, 0.25 * np.pi),
        "tangent_rotation": (0.05 * np.pi, 0.2 * np.pi),
        "change_binding_state": (0, 0)
    })
    return bead_amp_bounds, move_amp_bounds
