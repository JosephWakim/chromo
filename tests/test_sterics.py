"""Test Nucleosome Sterics.

These tests are designed to ensure that the identification of steric clashes
between nucleosomes is functioning as expected.
"""

import os
import sys
import time

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
parent_dir = cwd + "/.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

import numpy as np
from inspect import getmembers, isfunction

import chromo.polymers as poly
import chromo.mc.moves as mv
import chromo.mc as mc
from chromo.util.nucleo_geom import R_default
import chromo.binders
from chromo.fields import NullField
import chromo.util.mu_schedules as ms
from wlcstat.chromo import gen_chromo_conf

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())


def test_nucleosome_sterics():
    """Test the identification of steric clashes between nucleosomes.
    """

    # Define chromatin to match theory
    linker_length_bp = 36
    length_bp = 0.332
    linker_length = linker_length_bp * length_bp
    n_beads = 10
    linker_lengths = np.array([linker_length] * (n_beads-1))
    lp = 50.
    lt = 100.
    bp_wrap = 147.

    # Initialize the polymer
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )

    # Get the distances between pairs of nucleosomes
    p.get_distances()
    distances = p.distances.copy()
    distances_trial = p.distances_trial.copy()
    # Trial configuration should be equivalent to the initial configuration
    assert np.allclose(distances, distances_trial), \
        "Trial configuration should be equivalent to the initial configuration."

    # Verify that there are no steric clashes in the initial configuration
    n_clashes = p.check_steric_clashes(distances)
    assert n_clashes == 0, \
        "There should be no steric clashes in the initial configuration."
    n_clashes_trial = p.check_steric_clashes(distances_trial)
    assert n_clashes_trial == 0, \
        "There should be no steric clashes in the trial configuration."

    # Move a nucleosome to create a steric clash
    p.r[0, :] = p.r[1, :] + np.array([0.5, 0, 0])
    p.get_distances()
    distances = p.distances.copy()
    n_clashes = p.check_steric_clashes(distances)
    assert n_clashes == 1, \
        f"There should be 1 steric clash in the current configuration;" \
        f" there are {n_clashes}."
    n_clashes_trial = p.check_steric_clashes(distances_trial)
    assert n_clashes_trial == 0, \
        "There should be no steric clashes in the trial configuration."


def test_distance_runtime():
    """Get the average runtime for computing pairwise distances.

    We will check how long it takes to compute pairwise distances for a
    chromatin fiber with 100 nucleosomes.
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
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )

    # Determine average time to get pairwise distances
    n_trials = 100
    times = np.zeros(n_trials)
    for i in range(n_trials):
        start = time.time()
        p.get_distances()
        times[i] = time.time() - start

    print(f"Average time to evaluate get_distances(): {np.mean(times)} s")

    # Raise an error if the runtime is too long. This is not a strict test,
    # but it is useful to know if the code has become too inefficient.
    assert np.mean(times) < 0.01, \
        f"The average time to evaluate get_distances() has become too large; " \
        f"the average runtime is {np.mean(times)} s. Please reduce this time " \
        f"to less than 0.01 s."


def test_distances():
    """Check that the correct pairwise distances are being calculated.
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
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )
    # Compute pairwise distances
    p.get_distances()
    # Check pairwise distances
    distances = p.distances
    distances_trial = p.distances_trial
    for i in range(n_beads-1):
        assert np.isclose(distances[i, i+1], linker_length, atol=1e-10), \
            f"Distance between nucleosome {i} and {i+1} is incorrect"
        assert np.isclose(distances_trial[i, i+1], linker_length, atol=1e-10), \
            f"Distance between nucleosome {i} and {i+1} is incorrect"
    # Check all pairwise distances
    for i in range(n_beads-1):
        for j in range(i+2, n_beads):
            assert np.isclose(
                distances[i, j], (j - i) * linker_length, atol=1e-10
            ), f"Distance between nucleosome {i} and {j} is incorrect"
            assert np.isclose(
                distances_trial[i, j], (j - i) * linker_length, atol=1e-10
            ), f"Distance between nucleosome {i} and {j} is incorrect"


def evaluate_runtime_with_sterics():
    """Evaluate the runtime required to run a simulation with steric clashes
    """
    n_beads = 200
    linker_lengths_bp = np.ones(n_beads-1, dtype=int) * 15
    length_bp = 0.332
    linker_lengths = linker_lengths_bp * length_bp
    bp_wrap = 147.
    lp = 150.6024096385542 / length_bp
    lt = 301.2048192771084 / length_bp

    # Load chemical modifications
    chemical_mods = np.zeros((n_beads, 1), dtype=int)
    chemical_mods[:20, 0] = 2
    modification_name = "H3K9me3"

    # Instantiate the HP1 reader protein
    binder = chromo.binders.get_by_name('HP1')
    binder.chemical_potential = -9.7

    # Adjust the interaction distance to match that used in the nucleosome
    # positioning model
    interaction_radius = 8.0
    interaction_volume = (4.0/3.0) * np.pi * interaction_radius ** 3
    binder.interaction_radius = interaction_radius
    binder.interaction_volume = interaction_volume
    binders = chromo.binders.make_binder_collection([binder])

    # Binding states
    states = np.zeros(chemical_mods.shape, dtype=int)

    # Initialize the polymer
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        binders=[binder],
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["HP1"]),
        chemical_mods=chemical_mods,
        chemical_mod_names=np.array([modification_name])
    )

    # Update positions and orientations using chain growth algorithm
    _, _, _, rn, un, orientations = gen_chromo_conf(
        linker_lengths_bp.astype(int), return_orientations=True
    )
    t3_temp = orientations["t3_incoming"]
    t2_temp = orientations["t2_incoming"]
    p.r = rn.copy()
    p.r_trial = rn.copy()
    p.t3 = t3_temp.copy()
    p.t3_trial = t3_temp.copy()
    p.t2 = t2_temp.copy()
    p.t2_trial = t2_temp.copy()

    field = NullField()
    amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds([p])

    # Create a list of mu schedules, which are defined in another file
    schedules = [func[0] for func in getmembers(ms, isfunction)]
    select_schedule = "linear_step_for_negative_cp"
    mu_schedules = [
        ms.Schedule(getattr(ms, func_name)) for func_name in schedules
    ]
    mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]

    random_seed = np.random.randint(1, 100000)
    output_dir = "tests/test_output"

    path_to_run_script = os.path.abspath(__file__)

    # Determine average time to run an MC step
    n_trials = 20
    mc_steps_per_snapshot = n_trials
    n_snapshots = 1

    start = time.time()
    _ = mc.polymer_in_field(
        [p],
        binders,
        field,
        mc_steps_per_snapshot,
        n_snapshots,
        amp_bead_bounds,
        amp_move_bounds,
        output_dir=output_dir,
        mu_schedule=mu_schedules[0],
        random_seed=random_seed,
        path_to_run_script=path_to_run_script
    )
    runtime = time.time() - start
    print(f"Average time per MC move: {runtime / n_trials} s")

    # Raise an error if the runtime is too long. This is not a strict test,
    # but it is useful to know if the code has become too inefficient.
    assert np.mean(runtime / n_trials) < 6.5, \
        f"The average time to run an MC step has become too large; " \
        f"the average runtime is {runtime / n_trials} s. Please reduce this " \
        f"time to less than 6.5 s."


def test_cutoff_distance():
    """Test that the nucleosome radius is stored correctly.

    Notes
    -----
    The nucleosome radius is used to assess steric clashes.
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
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )
    # Load the nucleosome radius
    nucleosome_radius = p.beads[0].rad
    assert np.isclose(nucleosome_radius, R_default, atol=1e-10), \
        f"The nucleosome radius is incorrect; it is {nucleosome_radius} " \
        f"instead of {R_default}."


def test_steric_clash_compute_dE():
    """Test that the steric clash energy is computed correctly.
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
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )

    # Verify that the number of beads is measured correctly
    n_beads_check = p.num_beads
    assert n_beads_check == n_beads, \
        f"The number of beads is incorrect; it is {n_beads_check} instead " \
        f"of {n_beads}."
    assert len(p.r) == n_beads, \
        f"The number of bead positions is incorrect; it is {len(p.r)} " \
        f"instead of {n_beads}."

    # Verify that distances are zero on the main diagonal
    for i in range(n_beads):
        assert np.isclose(p.distances[i, i], 0), \
            f"The distance between bead {i} and itself is not zero; it is " \
            f"{p.distances[i, i]}."
        assert np.isclose(p.distances_trial[i, i], 0), \
            f"The trial distance between bead {i} and itself is not zero; " \
            f"it is {p.distances_trial[i, i]}."

    # The initial energy should not be so large
    E_tot = p.compute_E()
    assert E_tot < 1E25, f"The initial energy is too large; it is {E_tot}."

    # Move trial configuration to a position that causes a steric clash
    p.r_trial[1] = p.r_trial[0].copy() + np.array([1.8 * R_default, 0, 0])
    # Compute the change in energy associated with the move
    p.get_delta_distances(1, 2)

    print(f"Original Distance: {round(np.asarray(p.distances)[0, 1], 3)}")
    print(f"New Distance: {round(np.asarray(p.distances_trial)[0, 1], 3)}")
    print(f"Excluded Distance: {round(p.excluded_distance, 3)}")
    assert np.isclose(p.distances_trial[0, 1], p.r_trial[1][0] - p.r[0][0]), \
        f"The distance between nucleosomes 0 and 1 is incorrect; it is " \
        f"{p.distances_trial[0, 1]} instead of {p.r_trial[1][0] - p.r[0][0]}."

    overlap_ratio = 2 * R_default / (p.r_trial[1][0] - p.r[0][0])
    print(f"Overlap Ratio: {round(overlap_ratio, 3)}")
    dE = p.eval_delta_steric_clashes(1, 2)
    # The energy change should be enormous, because of the steric clash
    overlap_ratio = 2 * R_default / (p.r_trial[1][0] - p.r[0][0])
    dE_expected = (overlap_ratio ** 12) - 2 * (overlap_ratio ** 6) + 1
    assert np.isclose(dE, dE_expected), \
        f"Expected steric energy change of {dE_expected}; it is {dE}."

    # Create multiple steric clashes and verify that the energy is still stable
    for i in range(1, n_beads):
        p.r_trial[i] = p.r_trial[i-1].copy()
        p.r_trial[i][0] += 0.999 * (p.beads[i].rad * 2)
    dE = p.compute_dE("slide", np.arange(n_beads), n_beads)

    p.get_distances()
    overlap_ratio = 2 * R_default / (p.r_trial[1][0] - p.r[0][0])
    steric_contribution = (overlap_ratio ** 12) - 2 * (overlap_ratio ** 6) + 1
    dE_expected = 0
    for i in range(1, p.num_beads):
        dE_expected += steric_contribution
    assert np.isclose(dE, dE_expected), f"The energy change is unstable; " \
        f"it is represented as {dE} instead of the expected value of " \
        f"{dE_expected}."

    # We did not introduce binders, so the total interaction energy should be 0
    E_dict = p.compute_E_detailed()
    assert np.isclose(E_dict["interaction"], 0), \
        "The total interaction energy should be 0."


def test_update_distances():
    """Verify that the model correctly updates pairwise distances.
    """
    # Define chromatin to match theory
    linker_length_bp = 36
    length_bp = 0.332
    linker_length = linker_length_bp * length_bp
    n_beads = 100
    linker_lengths = np.array([linker_length] * (n_beads - 1))
    lp = 50.
    lt = 100.
    bp_wrap = 147.

    # Initialize the polymer
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )
    move = mv.slide
    adaptive_move = mv.MCAdapter(
        'test_output/acceptance_trackers',
        move.__name__ + "_snap_",
        move,
        moves_in_average=20,
        init_amp_bead=0,
        init_amp_move=100
    )

    # Test Move Acceptance
    for i in range(1, n_beads-5):
        p.r_trial[i] = p.r_trial[i - 1].copy()
        p.r_trial[i][0] += 0.999 * (p.beads[i].rad * 2)
    p.get_distances()
    dist = (p.r_trial[1][0] - p.r_trial[0][0])
    assert isinstance(p, poly.DetailedChromatinWithSterics), \
        "The polymer is not of the correct type."
    adaptive_move.accept(
        p, 0, np.arange(p.num_beads), p.num_beads, log_move=False,
        log_update=False, update_distances=True
    )
    for i in range(1, n_beads-5):
        assert np.isclose(p.r[i][0] - p.r[i-1][0], dist), \
            "Error updating distances when accepting move."
    for i in range(0, n_beads):
        for j in range(0, n_beads):
            assert np.isclose(p.distances[i, j], p.distances_trial[i, j]), \
                "Current and trial distances are out of sync."

    # Test Move Rejection
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )
    for i in range(1, n_beads-5):
        p.r_trial[i] = p.r_trial[i - 1].copy()
        p.r_trial[i][0] += 0.999 * (p.beads[i].rad * 2)
    p.get_distances()
    dist = (p.r[1][0] - p.r[0][0])
    assert isinstance(p, poly.DetailedChromatinWithSterics), \
        "The polymer is not of the correct type."
    adaptive_move.reject(
        p, 0, np.arange(p.num_beads), p.num_beads, log_move=False,
        log_update=False, update_distances=True
    )
    for i in range(1, n_beads-5):
        assert np.isclose(p.r_trial[i][0] - p.r_trial[i-1][0], dist), \
            "Error updating distances when rejecting move."
    for i in range(0, n_beads):
        for j in range(0, n_beads):
            assert np.isclose(p.distances[i, j], p.distances_trial[i, j]), \
                "Current and trial distances are out of sync."

    # We did not touch the states. Verify that they are all zero
    assert np.allclose(np.asarray(p.states), 0), \
        "The states should all be zero."
    assert np.allclose(np.asarray(p.states_trial), 0), \
        "The trial states should all be zero."

    # The binding energy and interaction energies should be zero
    dE_bind = p.get_E_bind()
    assert np.isclose(dE_bind, 0), \
        "The binding energy should be zero."
    dE_interact = p.evaluate_binder_interactions()
    assert np.isclose(dE_interact, 0), \
        "The interaction energy should be zero."


def test_compute_dE_sterics():
    """Check the calculation of the energy change for a move.
    """
    # Define chromatin to match theory
    linker_length_bp = 36
    length_bp = 0.332
    linker_length = linker_length_bp * length_bp
    n_beads = 100
    linker_lengths = np.array([linker_length] * (n_beads - 1))
    lp = 50.
    lt = 100.
    bp_wrap = 147.

    # Initialize the polymer
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )
    move = mv.slide
    adaptive_move = mv.MCAdapter(
        'test_output/acceptance_trackers',
        move.__name__ + "_snap_",
        move,
        moves_in_average=20,
        init_amp_bead=0,
        init_amp_move=100
    )

    # Evaluate energy of initial configuration
    E_dict = p.compute_E_detailed()
    E_total_original = E_dict["total"]
    E_elastic_original = E_dict["elastic"]
    E_interaction_original = E_dict["interaction"]
    E_steric_original = E_dict["steric"]
    print(f"Original Clash Energy: {E_steric_original}")
    print(f"Original Elastic Energy: {E_elastic_original}")
    assert np.isclose(E_steric_original, p.eval_E_steric_clashes()), \
        "The initial steric energy is not being calculated correctly."
    assert np.isclose(E_interaction_original, 0), \
        "The interaction energy should be zero."

    # Move nucleosomes to generate clashes
    dE = 0
    dE_clash = 0
    for i in range(1, n_beads):

        # Evaluate the starting energy of the move
        # E_dict_start = p.compute_E_detailed()

        # Conduct a move and evaluate the change in energy
        p.r_trial[i] = p.r_trial[i - 1].copy()
        p.r_trial[i][0] += 0.999 * (p.beads[i].rad * 2)
        p.get_delta_distances(i, i+1)
        dE_clash_ = p.eval_delta_steric_clashes(i, i+1)
        dE_clash += dE_clash_
        inds = np.array([i])
        dE_ = p.compute_dE("slide", inds, 1)
        dE += dE_
        # "accept" the move and evaluate final steric energy
        adaptive_move.accept(
            p, 0, inds, 1, log_move=False,
            log_update=False, update_distances=True
        )

        # Evaluate the final energy of the move and verify consistency
        # E_dict_end = p.compute_E_detailed()
        # assert np.isclose(
        #   E_dict_end["steric"], E_dict_start["steric"] + dE_clash_
        # ), "The steric energy change is not consistent."
        # assert np.isclose(
        #     E_dict_end["elastic"], E_dict_start["elastic"] + (dE_-dE_clash_)
        # ), "The elastic energy change is not consistent."
        # assert np.isclose(E_dict_end["total"], E_dict_start["total"] + dE_), \
        #     "The energy change is not consistent."

    # Check calculation of steric energy
    E_dict = p.compute_E_detailed()
    E_steric_final_1 = p.eval_E_steric_clashes()
    E_steric_final_2 = E_dict["steric"]
    print("Final Energy Total:", E_dict["total"])
    print("Final Steric Energy:", E_steric_final_1)
    assert np.isclose(E_steric_final_1, E_steric_final_2), \
        "The final steric energy is not being calculated consistently."
    assert np.isclose(E_steric_final_1, E_steric_original + dE_clash), \
        "Change in steric energy is not consistent."

    # Check the total energy
    E_total_final = E_dict["total"]
    assert(np.isclose(E_total_final, E_total_original + dE)), \
        "The total energy is not being calculated consistently."

    # Check calculation of elastic energy
    E_elastic_final_1 = E_dict["elastic"]
    E_elastic_final_2 = E_dict["total"] - E_steric_final_2
    print("Final Elastic Energy:", E_elastic_final_1)
    assert np.isclose(E_elastic_final_1, E_elastic_final_2), \
        "The final elastic energy is not being calculated consistently."
    assert np.isclose(E_elastic_original + (dE-dE_clash), E_elastic_final_1), \
        "Change in elastic energy is inconsistent."


if __name__ == "__main__":
    test_nucleosome_sterics()
    test_distance_runtime()
    evaluate_runtime_with_sterics()
    test_distances()
    test_cutoff_distance()
    test_steric_clash_compute_dE()
    print("All tests passed.")
