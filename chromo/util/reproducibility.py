"""A module with routines designed to make code "automatically" reproducible.

Reproducible here should be taken in the scientific sense. We want to, given
the output of our simulation, always be able to
    1. Reconstruct the inputs to the simulation exactly.
    2. Run our code again with those inputs to generate that exact same output.

For our purposes, this means saving
    1. The Git commit hash of the simulation used.
    2. All random seeds used.
    3. Serializing all inputs used.

If in doubt, you probably want to simply wrap any simulation function you write
with :ref:`make_reproducible`.

Created by Bruno Beltran; Adapted by Joseph Wakim
"""

from pathlib import Path
import inspect
import collections
from typing import List, Callable, Any, Tuple, Dict

import pandas as pd
import numpy as np

# get version number (includes commit hash) from versioneer
from .._version import get_versions

__version__ = get_versions()['version']
del get_versions

# enforce a uniform kwarg for output directory
output_dir_param_name = 'output_dir'
# output_dir_param_name / sim_tracker_name will store history of simulations
# run in that output directory
sim_tracker_name = Path('simulations.csv')
# the actual simulation output will go in
sim_folder_prefix = Path('sim_')


class make_reproducible(object):
    f"""Decorator logs similation parameters for reproducibility.

    Requires all arguments of the original function to implement the `.to_file`
    method and `.name` attribute. Saves the state of the simulation's input to
    the output folder. The required keyword argument {output_dir_param_name}
    and a required random seed kwarg `random_seed` must also be specified.

    The versioneer version (including a commit hash if necessary), the
    date/time, a unique folder name where the simulation inputs and output will
    be stored, the output directory and file name from which the simulation is
    continuing, and any arguments to the function with built-in types will be
    appended as a row to the {sim_tracker_name} file, for easy opening as a
    Pandas DataFrame tracking all simulations done in a given output directory.

    The unique output directory name will be forwarded into the simulation
    function via the {output_dir_param_name} kwarg, meaning that simulation
    output will typically be in a subfolder of the requested folder to prevent
    overwriting old output.

    The final directory structure, if you request

        >>> {output_dir_param_name} = "output_dir"

    at run time, will look like:

        output_dir
        ├── {sim_tracker_name}
        ├── {sim_folder_prefix}1
        │   ├── param1
        │   ├── ...
        │   ├── paramN
        │   └── (Any output of simulator goes here)
        ├── ...
        └── {sim_folder_prefix}M
            ├── param1
            ├── ...
            ├── paramN
            └── (Any output of simulator goes here)
    """

    def __init__(self, sim: Callable[[Any], Any]):
        """Initialize the `make_reproducible` decorator object

        Parameters
        ----------
        sim : Callable[[Any], Any]
            Simulation function being wrapped.
        """
        self.sim = sim
        self.params = inspect.signature(sim).parameters
        self.outdir_param = self.params[output_dir_param_name]

    def __call__(self, *args, **kwargs):
        """Run the wrapper on simulation function.

        Begins by checking that all the required parameters (a random seed and
        an output directory) are specified in the simulation function. Then it
        verifies that the output directory is either a kwarg or has a default
        value. Then gets a new folder to save the simulation in a thread-safe
        way, creating a new base output directory if it does not exist.

        After creating the new output directory, separates the list of
        parameters into those which are simple to log into a CSV file and those
        which are more complicated. We then add the simulation version number
        and a timestamp, which are always good to log.

        The simple logged components are appended to the `sim_tracker_name`
        output file, if such a file exists. Otherwise, the file is created with
        an appropriate header row.

        The hard-to-log parameters are logged to separate files using the
        `to_file_params` method. These parameters are written to the output
        file using their `to_file` methods.
        
        Finally, run the function and log the completion time.
        """
        self.check_required_parameters()
        kwargs = self.check_output_directory(**kwargs)

        outdir = Path(kwargs[output_dir_param_name])
        outdir.mkdir(parents=True, exist_ok=True)
        output_subfolder = get_unique_subfolder(outdir / sim_folder_prefix)
        kwargs[output_dir_param_name] = output_subfolder

        self.log_parameters(outdir, output_subfolder, *args, **kwargs)

        sim_out = self.sim(*args, **kwargs)
        with open(output_subfolder / Path('__completion_time__'), 'w') as f:
            f.write(str(pd.Timestamp.now()))
        return sim_out

    def log_parameters(self, outdir, output_subfolder, *args, **kwargs):
        """Log parameters used to call simulation.

        `simple_params` are parameters with a single value that can be stored
        in an aggregated column for all simulations. `hard_params` are
        parameters that require more than one value to record and are stored in
        separate files.

        When a log of simple parameters exists containing aggregated simulation
        data, the log must be checked to make sure it contains columns for all
        values in `simple_params`. If the log contains the proper columns,
        `simple_params` is appended to the end of the log file. If the log does
        not contain proper columns, then the log is reconstructed with
        `simple_params` added to the end of it.

        Parameters
        ----------
        outdir : Path
            Path to output directory for all simulations
        output_subfolder : Path
            Path to the output directory for the specific simulation being run
        """
        simple_params, hard_params = split_builtin_params(
            self.sim, *args, **kwargs
        )
        simple_params['version'] = __version__
        simple_params['start_time'] = pd.Timestamp.now()
        sim_tracker = outdir / sim_tracker_name
        simple_params = pd.DataFrame(pd.Series(simple_params)).T
        log_missing = not sim_tracker.exists()
        mode = 'a'
        if not log_missing:
            tracker = pd.read_csv(sim_tracker, nrows=0)
            if not all(
                item in tracker.columns.tolist()
                for item in simple_params.columns.tolist()
            ):
                tracker = pd.read_csv(sim_tracker)
                mode = 'w'
                log_missing = True
            simple_params = pd.concat([tracker, simple_params], axis=0)
        simple_params.to_csv(
            sim_tracker, header=log_missing, index=False, mode=mode
        )
        to_file_params(hard_params, output_subfolder)

    def check_required_parameters(self):
        f"""All simulations need a random seed and an output directory.

        The random seed is stored in the `random_seed` kwarg, while the output
        directory is stored in the {output_dir_param_name} kwarg.
        """
        if "random_seed" not in self.params:
            raise ValueError(
                "Requires the random seed to be passed in explicitly, even if the "
                "random generator is passed to the function as well."
            )
        if output_dir_param_name not in self.params:
            raise ValueError(
                f"Could not wrap simulation function: {self.sim}.\n"
                f"Missing required kwarg: {output_dir_param_name}."
            )

    def check_output_directory(self, **kwargs):
        """Check that the output directory is specified or contains a default value.
        """
        if output_dir_param_name not in kwargs:
            if self.outdir_param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Could not run simulation function: {self.sim}.\n"
                    f"{output_dir_param_name} not specified, and "
                    "no default value can be found."
                )
            kwargs[output_dir_param_name] = self.outdir_param.default
        return kwargs


builtin_types = [bool, bytes, bytearray, complex, float, int, str]
"""
Types to be saved as CSV entries by default.

These are basically just the built-ins that permit simple save/load cycling via
CSV entries with Pandas. Some extras are included in the split_builtin_params
function itself. Please see that code for a full description.

The following strategy would also work, but would lead to us having to pickle
some builtin things, like "map", "Error", etc..
builtin_types = tuple(getattr(builtins, t) for t in dir(builtins)
    if isinstance(getattr(builtins, t), type))

These additional types are not allowed by default because they make loading in
the CSV a little trickier:
..., frozenset, tuple]
"""


def split_builtin_params(
    sim: Callable[[Any], Any], *args, **kwargs
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split parameters between built in and non-built in data types.

    Parameters
    ----------
    sim : Callable[[Any], Any]
        Any function on which `make_reproducible` is being called.

    Returns
    -------
    Dict[str, Any]
        Dictionary of parameters for built in data types
    Dict[str, Any]
        Dictionary of parameters for non-built in data types
    """
    sig = inspect.signature(sim).bind(*args, **kwargs)
    builtin_args = {}
    non_builtin = {}
    for arg_name, value in sig.arguments.items():
        keys, values = parse_type(arg_name, value)
        for i in range(len(keys)):
            dtype = type(values[i])
            if dtype in builtin_types or issubclass(dtype, Path):
                builtin_args[keys[i]] = values[i]
            else:
                non_builtin[keys[i]] = values[i]
    return builtin_args, non_builtin


def parse_type(
    arg_name: str, value: Any
) -> Tuple[List[str], List[Any]]:
    """Parse argument data types (for handling kwargs)

    Parameters
    ----------
    arg_name : str
        Name of the argument
    value : Any
        Value associated with `arg_name`

    Returns
    -------
    List[str]
        List of argument names after parsing
    List[Any]
        List of argument values after parsing
    """
    dtype = type(value)
    if dtype is dict:
        keys = []
        values = []
        for key, val in value.items():
            sub_keys, sub_vals = parse_type(key, val)
            keys += sub_keys
            values += sub_vals
        return keys, values
    return [arg_name], [value]


def get_unique_subfolder(root):
    """
    Create and return a thread-safe unique folder.

    Parameters
    ----------
    root : str or Path
        The base of the filename which we should try appending numbers to until
        we find a number which doesn't already exist.

    Returns
    -------
    Path
        The folder which was created.

    Notes
    -----
    Uses the fact that mkdir is required to be atomic on POSIX-compliant system
    to make sure that two threads aren't given the same folder.
    """
    i = 1
    while True:
        folder_name = Path(str(root) + str(i))
        try:
            folder_name.mkdir(parents=True)
            break
        except:
            i += 1
    return folder_name


def to_file_params(non_builtins_kwargs, folder, suffix=''):
    """
    Call ``.to_file`` for each parameter, handling sequences.

    Parameters
    ----------
    non_builtins_kwargs : Dict[str, RoundTrippable]
        A RoundTrippable must implement the
        ``.name``/``.to_file``/``.from_file`` interface, or be a standard type,
        such as a numpy array or a DataFrame.
    folder : Path
        The folder to save the output to.
    """
    for arg_name, value in non_builtins_kwargs.items():
        dtype = type(value)
        if isinstance(value, collections.Sequence):
            for data in value:
                to_file_params({arg_name: data}, folder, suffix)
        elif hasattr(value, 'to_file'):
            value.to_file(folder / Path(value.name + suffix))
        elif issubclass(dtype, pd.DataFrame) or issubclass(dtype, pd.Series):
            value.to_csv(folder / Path(arg_name + suffix))
        elif issubclass(dtype, np.ndarray):
            np.savetxt(folder / Path(arg_name + suffix), value)
        else:
            raise ValueError(f"Argument not understood: {arg_name}={value}.\n"
                             "This simulation cannot be made reproducible.\n"
                             f"Please implement `.to_file` for type: {dtype}.")
