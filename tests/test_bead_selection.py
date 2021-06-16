"""Test bead selection from left

For all tests, use a selection window size of 10,000 beads.
"""

import chromo.util.bead_selection as beads

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, anderson_ksamp


def plot_distribution_of_bead_selections(
    selections: np.ndarray,
    discrete_bins: np.ndarray,
    x_theoretical: np.ndarray,
    y_theoretical: np.ndarray,
    x_axis_label: str,
    file_name: str
):
    """Plot distribution of expected/observed bead selections.

    Parameters
    ----------
    selections : np.array
        1D array of bead selection indices from sample
    discrete_bins : np.array
        1D array of bin edges between which selections are distributed
    x_theoretical : np.array
        1D array of all bead indices
    y_theoretical : np.array
        1D array of probabilities corresponding to indices in x_theoretical
    x_axis_label : str
        Label to put on x-axis of plot
    file_name : str
        Path from cwd to save file
    """
    plt.hist(selections, bins=discrete_bins, density=True)
    plt.plot(x_theoretical, y_theoretical)
    plt.xlabel(x_axis_label)
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def batch_bead_selection(
    num_beads: int,
    select_from_left: bool = True,
    ind0: int = 0
):
    """Draw 10000 beads (with repeats) from an exponential distribution.

    Parameters
    ----------
    num_beads : int
        Number of beads in polymer
    select_from_left : boolean
        True (default) if exponentially selecting beads from LHS of polymer;
        False if selecting from RHS; None if from point
    ind0 : int
        If selecting a bead from the middle of a distribution, this is the
        central bead in the two sided exponential distribution.

    Returns
    -------
    selections : np.array
        1D array of bead indices exponentially selected
    select_densities : np.array
        1D array of bead selection probabilities within each bin
    select_bin_edges : np.array
        1D array of bins for bead selection probabilities
    """

    window = num_beads
    discrete_bins = np.arange(0, num_beads, 100)
    num_trials = 10000

    selections = np.zeros(num_trials)

    if select_from_left is None:    # Test rotation from point
        for i in range(num_trials):
            selections[i] = beads.select_bead_from_point(
                window, num_beads, ind0
            )

    elif select_from_left:          # Test LHS rotation
        for i in range(num_trials):
            selections[i] = beads.select_bead_from_left(
                window, num_beads
            )

    else:                           # Test RHS rotation
        for i in range(num_trials):
            selections[i] = beads.select_bead_from_right(
                window, num_beads
            )

    select_densities, select_bin_edges = np.histogram(
        selections,
        bins=discrete_bins,
        density=True
    )

    return selections, select_densities, select_bin_edges


def compare_with_expected_selections_ends(
    num_beads,
    select_densities,
    select_bin_edges,
    select_from_left=True
):
    """Compare randomly selected beads with an exponential distribution.

    Parameters
    ----------
    num_beads : int
        Number of beads in polymer
    select_densities : np.array
        1D array of bead selection probabilities within each bin
    select_bin_edges : np.array
        1D array of bins for bead selection probabilities
    select_from_left : boolean
        True (default) if exponentially selecting beads from LHS of polymer;
        False if selecting from RHS

    Returns
    -------
    x : np.array
        1D array of all bead indices
    y : np.array
        1D array of probabilities corresponding to indices in x_theoretical
    float
        AD-test statistic comparing observed and expected distributions
    """
    eCDF_val_at_window = 0.99
    p = 1 - (1 - eCDF_val_at_window) ** (1 / (num_beads + 1))

    x = np.arange(0, num_beads, 1)
    pdf = expon.pdf(x, loc=0, scale=1/p)
    cdf = np.cumsum(pdf)
    y = pdf / (cdf[-1] - cdf[0])

    if not select_from_left:
        y = np.flip(y)

    y_discrete, _ = np.histogram(y, bins=select_bin_edges)

    return x, y, anderson_ksamp([select_densities, y_discrete])[0]


def compare_with_expected_selections_center(
    num_beads,
    select_densities,
    select_bin_edges,
    center_of_dist
):
    """Compare randomly selected beads w/ two-sided exponential distribution.

    Parameters
    ----------
    num_beads : int
        Number of beads in polymer
    select_densities : np.array
        1D array of bead selection probabilities within each bin
    select_bin_edges : np.array
        1D array of bins for bead selection probabilities
    center_of_dist : int
        Bead index at the center of the distribution

    Returns
    -------
    x : np.array
        1D array of all bead indices
    y : np.array
        1D array of probabilities corresponding to indices in x_theoretical
    float
        AD-test statistic comparing observed and expected distributions
    """
    eCDF_val_at_window = 0.99
    p_RHS = 1 - (1 - eCDF_val_at_window) ** (
        1 / (num_beads-center_of_dist + 1)
    )
    p_LHS = 1 - (1 - eCDF_val_at_window) ** (
        1 / (center_of_dist + 1)
    )

    # RHS
    x_RHS = np.arange(center_of_dist, num_beads, 1)
    pdf_RHS = expon.pdf(x_RHS, loc=center_of_dist, scale=1/p_RHS)
    cdf_RHS = np.cumsum(pdf_RHS)
    y_RHS = pdf_RHS / (cdf_RHS[-1] - cdf_RHS[0])

    # LHS
    x_LHS = np.arange(0, center_of_dist, 1)
    pdf_LHS = expon.pdf(x_LHS, loc=0, scale=1/p_LHS)
    cdf_LHS = np.cumsum(pdf_LHS)
    y_LHS = pdf_LHS / (cdf_LHS[-1] - cdf_LHS[0])
    y_LHS = np.flip(y_LHS)

    x = np.concatenate((x_LHS, x_RHS), axis=0)
    y = np.concatenate((y_LHS, y_RHS), axis=0)
    y = y / np.sum(y)

    y_discrete, _ = np.histogram(y, bins=select_bin_edges)

    return x, y, anderson_ksamp([select_densities, y_discrete])[0]


def test_select_bead_from_left():
    """Test for exponentially distributed bead selection from LHS.

    Trial runs turn up an AD-test statistic close to 175.

    For selection from left, first bead has higher probability than last bead.
    """
    num_beads = 10000
    selections, select_densities, select_bin_edges = batch_bead_selection(
        num_beads
    )
    x, y, AD_stat = compare_with_expected_selections_ends(
        num_beads, select_densities, select_bin_edges
    )

    plot_distribution_of_bead_selections(
        selections,
        select_bin_edges,
        x,
        y,
        "Index Selection from Left",
        "tests/plots/Exponential_Sampling_from_Left.PNG"
    )

    assert AD_stat < 180
    assert y[0] > y[-1]


def test_select_bead_from_right():
    """Test for exponentially distributed bead selection from RHS.

    Trial runs turn up an AD-test statistic close to 175.

    For selection from right, last bead has higher probability than first bead.
    """
    num_beads = 10000
    selections, select_densities, select_bin_edges = batch_bead_selection(
        num_beads, select_from_left=False
    )
    x, y, AD_stat = compare_with_expected_selections_ends(
        num_beads, select_densities, select_bin_edges, select_from_left=False
    )

    plot_distribution_of_bead_selections(
        selections,
        select_bin_edges,
        x,
        y,
        "Index Selection from Right",
        "tests/plots/Exponential_Sampling_from_Right.PNG"
    )

    assert AD_stat < 180    # Repeated trials produce values near 175
    assert y[0] < y[-1]


def test_select_bead_from_point():
    """Test for exponentially distributed bead selection from point.

    AD test statistics from repeated trials fluctuate quite a bit, but never
    exceed ~175.
    """
    num_beads = 10000
    pass_test = []

    for i in range(100):
        ind0 = np.random.randint(0, num_beads)
        selections, select_densities, select_bin_edges = batch_bead_selection(
            num_beads, select_from_left=None, ind0=ind0
        )
        x, y, AD_stat = compare_with_expected_selections_center(
            num_beads, select_densities, select_bin_edges, ind0
        )

        if (i+1) % 10 == 0:
            plot_distribution_of_bead_selections(
                selections,
                select_bin_edges,
                x,
                y,
                "Index Selection from Point: " + str(ind0),
                "tests/plots/Exponential_Sampling_from_Center_"+str(i+1)+".PNG"
            )
        pass_test.append(AD_stat < 180)

    assert np.all(pass_test)
