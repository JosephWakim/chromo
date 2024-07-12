from io import StringIO
import os

from pandas.testing import assert_frame_equal
import numpy as np

from chromo.polymers import Chromatin, DetailedChromatinWithSterics


def test_polymer_saving():
    p = Chromatin.straight_line_in_x(
        'Chr-1', np.ascontiguousarray(np.array([1.] * 9), dtype=np.float64),
        states=np.zeros((10, 1), dtype=int),
        binder_names=np.array(['HP1']),
        chemical_mods=np.zeros((10, 1), dtype=int),
        chemical_mod_names=np.array(['H3K9me3'])
    )
    df = Chromatin.from_file(StringIO(p.to_file(None)), 'Chr-1').to_dataframe()
    df_pre_round_trip = p.to_dataframe()
    assert_frame_equal(df, df_pre_round_trip)


def test_detailed_chromatin_with_sterics_saving():
    """Verify that we can save/load snapshots of detailed chromatin w/ sterics.
    """

    # Define chromatin to match theory
    linker_length_bp = 36
    length_bp = 0.332
    linker_length = linker_length_bp * length_bp
    n_beads = 100
    linker_lengths = np.array([linker_length] * (n_beads-1))
    lp = 50.
    lt = 100.
    bp_wrap = 147.

    # Initialize the polymer
    p = DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )

    # Compute the detailed energy of the configuration
    E_dict_init = p.compute_E_detailed()

    # Save a temporary file with the detailed chromatin configuration
    temp_save_file = "temp_test_components_save.csv"
    p.to_file(temp_save_file)

    # Save and load the configuration
    p_check = DetailedChromatinWithSterics.from_file(temp_save_file, 'Chr-1')

    # Compute the detailed energy of the new configuration
    E_dict_final = p_check.compute_E_detailed()

    # Verify that the energies are consistent
    for key in E_dict_init.keys():
        assert np.isclose(E_dict_init[key], E_dict_final[key]), \
            "Energies before and after loading DetailedChromatinWithSterics " \
            "are inconsistent."

    # Remove the temporary save file
    os.remove(temp_save_file)
