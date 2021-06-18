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

TODO: Update the `eval_end_to_end_dist.py` file to reflect this new
`poly_stat.py` module. This will involve instantiating polymer objects from
the output files.

TODO: Find a way of normalizing the polymer statistics by persistence length.
This will involve adding an attribute to the polymer object for persistence
length. This may be hard-coded for chroamtin, since it is a fixed value but
still must appear in an attribute.

Joseph Wakim
Spakowitz Lab
Modified: June 17, 2021
"""

# External Modules
import numpy as np

# Custom Modules
from chromo.polymers import Polymer


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
