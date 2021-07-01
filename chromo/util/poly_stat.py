"""Average Polymer Statistics

Generate average polymer statistics from Monte Carlo simulation output. This
module defines a `PolyStat` object, which loads polymer configurations from a
`Polymer` object. `PolyStat` can be used to sample beads from the polymer and
calculate basic polymer statistics such as mean squared end-to-end distance
and mean 4th power end-to-end distance.

Bead sampling methods include overlapping sliding windows and non-overlapping
sliding windows. Overlapping sliding windows offer the benefit of increased
data, though the results are biased by central beads which exist in multiple
bins of the average. Non-overlapping sliding windows reduce the bias in the
results, but include fewer samples in the average.

Joseph Wakim
Spakowitz Lab
Modified: June 17, 2021
"""
# Import built-in modules
import os
import sys
from typing import List, Optional

# External Modules
import matplotlib.pyplot as plt
import numpy as np

# Custom Modules
from chromo.polymers import Polymer, Chromatin


def overlap_sample(bead_separation, num_beads):
    """Generate list of bead index pairs for sliding window sampling scheme.

    Parameters
    ----------
    bead_separation : int
        Number of beads in window for average calculation
    num_beads : int
        Number of beads in the polymer chain

    Returns
    -------
    windows : array_like (N, 2)
        Pairs of bead indicies for windows of statistics
    """
    num_windows = num_beads - bead_separation
    windows = np.zeros((num_windows, 2))
    for i in range(num_windows):
        windows[i, 0] = i
        windows[i, 1] = i + bead_separation
    windows = windows.astype("int64")
    return windows


def jump_sample(bead_separation, num_beads):
    """Generate list of bead index pairs for non-overlaping sampling scheme.

    If the end of the polymer does not complete a bin, it is excluded from the
    average.

    Parameters
    ----------
    bead_separation : int
        Number of beads in window for average calculation
    num_beads : int
        Number of beads in the polymer chain

    Returns
    -------
    windows : array_like (N, 2)
        Pairs of bead indicies for windows of statistics
    """
    num_windows = int(np.floor(num_beads / bead_separation))
    windows = np.zeros((num_windows, 2))
    for i in range(num_windows):
        bin_start = i * bead_separation
        windows[i, 0] = bin_start
        windows[i, 1] = bin_start + bead_separation
    windows = windows.astype("int64")
    return windows


class PolyStats(object):
    """Class representation of the polymer statistics analysis toolkit.
    """

    sampling_options = ["overlap", "jump"]

    def __init__(self, polymer: Polymer, sampling_method: str):
        """Initialize the `PolyStats` object.

        Parameters
        ----------
        polymer : Polymer
            Object representing the polymer for which statistics are being
            evaluated
        sampling_method : str
            Bead sampling method, either "overlap" or "jump"
        """
        self.polymer = polymer

        if sampling_method in self.sampling_options:
            if sampling_method == "overlap":
                self.sample_func = overlap_sample
            else:
                self.sample_func = jump_sample
        else:
            raise ValueError(
                "Specified sampling method invalid. Method must be either \
                'overlap' or `jump` and is case sensitive."
            )

    def load_indices(self, bead_separation: int) -> np.ndarray:
        """Load bead indices for windows in the average.

        Parameters
        ----------
        bead_separation : int
            Separation of beads in windows for which average is calcualated.

        Returns
        -------
        np.ndarray (N, 2)
            Pairs of bead indicies for windows of statistics
        """
        num_beads = len(self.polymer.r)
        return self.sample_func(bead_separation, num_beads)

    def calc_r2(self, windows: np.ndarray) -> float:
        """Calculate the average squared end-to-end distance of the polymer.

        Mean squared end-to-end distance is non-dimensionalized by the
        persistence length of the polymer, dividing the dimensional quantity by
        ((2 * self.polymer.lp) ** 2).

        Parameters
        ----------
        windows : np.ndarray (N, 2)
            Windows of bead indices for which the average squared end-to-end
            distance will be calculated

        Returns
        -------
        float
            Average squared end-to-end distance for specified bead windows
        """
        N = windows.shape[0]
        r2 = 0
        for window in windows:
            r_start = self.polymer.r[window[0], :]
            r_end = self.polymer.r[window[1], :]
            r2 += np.linalg.norm(r_end - r_start) ** 2
        return r2 / N / ((2 * self.polymer.lp) ** 2)

    def calc_r4(self, windows: np.ndarray) -> float:
        """Calculate the average 4th power end-to-end distance of the polymer.

        Mean 4th power end-to-end distance is non-dimensionalized by the
        persistence length of the polymer, dividing the dimensional quantity by
        ((2 * self.polymer.lp) ** 4).

        Parameters
        ----------
        windows : np.ndarray (N, 2)
            Windows of bead indices for which the average 4th moment end-to-end
            distance will be calculated

        Returns
        -------
        float
            Average 4th power end-to-end distance for specified bead windows
        """
        N = windows.shape[0]
        r4 = 0
        for window in windows:
            r_start = self.polymer.r[window[0], :]
            r_end = self.polymer.r[window[1], :]
            r4 += np.linalg.norm(r_end - r_start) ** 4
        return r4 / N / ((2 * self.polymer.lp) ** 4)


def get_latest_simulation(dir_: Optional[str] = '.'):
    """Get the latest simulation file from dir_.

    Parameters
    ----------
    dir_ : Optional[str]
        Path to directory into which the latest simulation will be retrieved

    Return
    ------
    str
        Most recent simulation output directory.
    """
    all_items = os.listdir(dir_)
    simulations = [
        item for item in all_items if os.path.isdir(dir_ + "/" + item)
    ]
    sim_IDs = [int(sim.split("_")[-1]) for sim in simulations]
    latest_sim_ID = np.argmax(sim_IDs)
    return simulations[latest_sim_ID]


def find_polymers_in_output_dir(dir_: Optional[str] = '.') -> List[str]:
    """Obtain the number of polymers in `_dir`.

    Parameters
    ----------
    dir_ : Optional[str]
        Path to directory into which the number of polymers will be retrieved

    Return
    ------
    List[str]
        List of polymer names
    """
    all_items = os.listdir(dir_)
    files = [
        item for item in all_items if os.path.isfile(dir_ + "/" + item)
    ]
    configs = [
        file for file in files if file.startswith("Chr-")
        and not file.endswith(".csv")
    ]
    polymer_nums = np.unique(
        np.array(
            [config.split("-")[1] for config in configs]
        )
    )
    polymers = ["Chr-" + str(i) for i in polymer_nums]
    return polymers


def get_latest_configuration(
    polymer_prefix: Optional[str] = "Chr-1",
    dir_: Optional[str] = '.'
) -> str:
    """Identify file path to latest polymer configuration in output directory.

    Parameters
    ----------
    polymer_prefix : str
        Polymer identifier in a configuration file, preceeding snapshot name
    dir_ : str
        Path to the output directory containing polymer configuration files

    Returns
    -------
    str
        Path to the file containing the latest polymer configuration file
    """
    all_items = os.listdir(dir_)
    configs = [
        file for file in all_items if file.startswith(polymer_prefix)
        and file.endswith(".csv")
    ]
    snapshots = [int(config.split("-")[2].split(".")[0]) for config in configs]
    latest_snapshot = np.max(snapshots)
    return dir_ + "/" + polymer_prefix + "-" + str(latest_snapshot) + ".csv"


def list_output_files(
    output_dir: str,
    equilibration_steps: int
) -> List[str]:
    """List configuration files in output directory following equilibration.

    Begins by listing all files in the output directory, then filters list to
    configuration files, and finally selects configuration snapshots following
    equilibration.

    Parameters
    ----------
    output_dir : str
        Output directory containing simulated configuration files
    equilibration_steps : int
        Number of Monte Carlo steps to ignore due to equilibration

    Returns
    -------
    List[str]
        List of files in the output directory corresponding to configurations
        following equilibration.
    """
    output_files = os.listdir(output_dir)
    output_files = [
        f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
    ]
    snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
    sorted_snap = np.sort(np.array(snapshot))
    output_files = [f for _, f in sorted(zip(snapshot, output_files))]
    output_files = [
        output_files[i] for i in range(len(output_files))
        if sorted_snap[i] > equilibration_steps - 1
    ]
    return output_files


def calc_mean_r2(
    output_dir: str,
    output_files: List[str],
    window_size: int
) -> float:
    """Calculate the mean squared end-to-end distance from configuration files.

    Parameters
    ----------
    output_dir : str
        Path to the configuration output directory for a particular simulation
        being analyzed
    output_files : List[str]
        List of output files from which to generate polymer statistics
    window_size : int
        Spacing between beads for which to calculate average squared end-to-end
        distance
    """
    r2 = []
    for i, f in enumerate(output_files):
        if (i+1) % 10 == 0:
            print("Snapshot: " + str(i+1) + " of " + str(len(output_files)))
            print()
        output_path = output_dir + "/" + f
        polymer = Chromatin.from_file(output_path, name=f)
        poly_stat = PolyStats(polymer, "overlap")
        r2.append(
            poly_stat.calc_r2(
                    windows=poly_stat.load_indices(window_size)
                )
        )
    return np.average(r2)


def save_summary_statistics(
    window_sizes: List[int],
    polymer_stats: List[float],
    save_path: str
):
    """Save summary polymer statistics to an output file.

    Parameters
    ----------
    window_sizes : List[int]
        Sequence of window sizes for which summary statistics are reported.
    polymer_stats : List[float]
        Polymer statistics being saved
    save_path : str
        Path to file in which to save summary statistics
    """
    if len(window_sizes) != len(polymer_stats):
        raise ValueError(
            "The number of polymer statistics does not match the number of \
            window sizes being saved."
        )
    with open(save_path, "w") as output_file:
        output_file.write("window_size, polymer_stat\n")
        for i in range(len(window_sizes)):
            output_file.write('%s, %s\n' % (window_sizes[i], polymer_stats[i]))


if __name__ == "__main__":
    """Generate plots from polymer statistics.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    parent_dir = cwd + "/../.."
    sys.path.insert(1, parent_dir)

    sim = get_latest_simulation()
    output_dir = parent_dir + "/output/" + sim
    num_equilibration = 90

    log_vals = np.arange(-2, 3, 0.05)
    bead_range = 10 ** log_vals * 80
    bead_range = bead_range.astype(int)
    bead_range = np.array(
        [bead_range[i] for i in range(len(bead_range)) if bead_range[i] > 0]
    )
    bead_range = np.unique(bead_range)
    average_squared_e2e = np.zeros((len(bead_range), 1))
    window_sizes = []

    for j, window_size in enumerate(bead_range):
        if not window_size > 0:
            continue
        window_sizes.append(window_size)
        print("!!!!! WINDOW SIZE: " + str(window_size) + " !!!!!")
        average_squared_e2e[j] = calc_mean_r2(
            output_dir,
            list_output_files(output_dir, num_equilibration),
            window_size
        )

    save_summary_statistics(
        window_sizes,
        average_squared_e2e,
        output_dir + "/avg_squared_e2e.csv"
    )

    plt.figure()
    plt.scatter(window_sizes, average_squared_e2e)
    plt.xlabel(r"$L/(2l_p)$")
    plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(output_dir + "/Squared_e2e_vs_dist_v2.png", dpi=600)

    # Plot the mean squared end-to-end distance on a log-log plot
    plt.figure()
    plt.scatter(np.log10(window_sizes), np.log10(average_squared_e2e))
    plt.xlabel(r"Log $L/(2l_p)$")
    plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
    r2_theory = 2 * (
        bead_range / 2 - (1 - np.exp(-(2) * bead_range)) / (2) ** 2
    )
    plt.plot(np.log10(bead_range), np.log10(r2_theory))
    plt.savefig(output_dir + "/Log_Log_Squared_e2e_vs_dist_v2.png", dpi=600)
