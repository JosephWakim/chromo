"""Test that polymers converge to the correct bead spacing.

This test will run a short simulation with just 1000 MC moves on a homopolymer
with 25 beads. The model that we will used is DetailedChromatin, which is a
kinked wormlike chain. After the short simulation, we will check that the
linker lengths fluctuate around the specified setpoint.
"""

import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
parent_dir = cwd + "/.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

import numpy as np
import matplotlib.pyplot as plt
import chromo.mc as mc
import chromo.polymers as poly
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name
from wlcstat.chromo import gen_chromo_conf

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())


def test_linker_lengths():
    """Test that linker lengths fluctuate around the setpoint.
    """
    # Define chromatin to match theory
    linker_length_bp = 36
    length_bp = 0.332
    linker_length = linker_length_bp * length_bp
    n_beads = 400
    linker_lengths = np.array([linker_length] * (n_beads-1))
    lp = 50.
    lt = 100.
    bp_wrap = 147.

    # Instantiate the null reader protein
    # This is pre-defined in the `chromo.binders` module
    null_binder = chromo.binders.get_by_name('null_reader')

    # Create the binder collection
    binders = chromo.binders.make_binder_collection([null_binder])

    # Initialize the polymer
    p = poly.DetailedChromatin.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )
    linker_lengths_bp = np.ones(n_beads-1, dtype=int) * linker_length_bp
    _, _, _, rn, un, orientations = gen_chromo_conf(
        linker_lengths_bp, lp=lp/length_bp, lt=lt/length_bp,
        return_orientations=True
    )
    print(
        f"Initial Nucleosome Spacing = "
        f"{np.linalg.norm(rn[1:] - rn[:-1], axis=1)}"
    )
    p.r = rn.copy()
    p.r_trial = rn.copy()
    p.t3 = orientations["t3_incoming"].copy()
    p.t3_trial = orientations["t3_incoming"].copy()
    p.t2 = orientations["t2_incoming"].copy()
    p.t2_trial = orientations["t2_incoming"].copy()

    # Load the linker lengths (pre-simulation)
    linker_lengths = []
    for i in range(n_beads-1):
        # Verify that the orientations are valid
        assert np.isclose(np.linalg.norm(p.t3[i]), 1)
        assert np.isclose(np.linalg.norm(p.t2[i]), 1)
        assert np.isclose(np.dot(p.t3[i], p.t2[i]), 0)
        ri_0, ro_0, t3i_0, t3o_0, t2i_0, t2o_0 = \
            p.beads[i].update_configuration(
                p.r[i, :], p.t3[i, :], p.t2[i, :]
            )
        ri_1, ro_1, t3i_1, t3o_1, t2i_1, t2o_1 = \
            p.beads[i+1].update_configuration(
                p.r[i+1, :], p.t3[i+1, :], p.t2[i+1, :]
            )
        link = np.linalg.norm(ri_1 - ro_0)
        linker_lengths.append(link)
    avg_linker_lengths_pre = np.mean(linker_lengths)
    print(f"All Linker Lengths: {linker_lengths}")
    print(f"Average Linker Length: {avg_linker_lengths_pre}")
    print(f"Extended Linker Length: {linker_length}")

    # Plot the initial linker length distribution
    # TODO: Plot -log Probabilities
    plt.figure(figsize=(6,3), dpi=300)
    plt.hist(linker_lengths, bins=20, color='blue', alpha=0.7)
    plt.axvline(linker_length, color='red', linestyle='--', label='Extended')
    plt.axvline(
        avg_linker_lengths_pre, color='black', linestyle='-', label='Average'
    )
    plt.xlabel('Linker Length (nm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Initial Linker Length Distribution')
    plt.tight_layout()
    plt.savefig('tests/plots/initial_linker_length_distribution.png', dpi=300)

    n_bins_x = 63
    n_bins_y = n_bins_x
    n_bins_z = n_bins_x

    x_width = 1000
    y_width = x_width
    z_width = x_width

    udf = UniformDensityField(
        polymers=[p],
        binders=binders,
        x_width=x_width,
        nx=n_bins_x,
        y_width=y_width,
        ny=n_bins_y,
        z_width=z_width,
        nz=n_bins_z,
        chi=0.0
    )

    amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers=[p])

    # Create a test output directory, if it does not exist
    output_dir = "tests/test_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    latest_sim = get_unique_subfolder_name(f"{output_dir}/sim_")
    moves_to_use = ctrl.all_moves_except_binding_state(
        log_dir=latest_sim,
        bead_amp_bounds=amp_bead_bounds.bounds,
        move_amp_bounds=amp_move_bounds.bounds,
        controller=ctrl.SimpleControl
    )

    num_snapshots = 2
    mc_steps_per_snapshot = 1000

    # Select a random seed
    random_seed = np.random.randint(1, 999999)

    p_sim = mc.polymer_in_field(
        polymers=[p],
        binders=binders,
        field=udf,
        num_save_mc=mc_steps_per_snapshot,
        num_saves=num_snapshots,
        bead_amp_bounds=amp_bead_bounds,
        move_amp_bounds=amp_move_bounds,
        output_dir=output_dir,
        random_seed=random_seed,
        mc_move_controllers=moves_to_use
    )

    # Load the linker lengths
    p_sim = p_sim[0]
    linker_lengths = []
    for i in range(n_beads-1):
        ri_0, ro_0, t3i_0, t3o_0, t2i_0, t2o_0 = \
            p_sim.beads[i].update_configuration(
                p_sim.r[i, :], p_sim.t3[i, :], p_sim.t2[i, :]
            )
        ri_1, ro_1, t3i_1, t3o_1, t2i_1, t2o_1 = \
            p_sim.beads[i+1].update_configuration(
                p_sim.r[i+1, :], p_sim.t3[i+1, :], p_sim.t2[i+1, :]
            )
        link = np.linalg.norm(ri_1 - ro_0)
        linker_lengths.append(link)
    avg_linker_lengths_post = np.mean(linker_lengths)
    print(f"All Linker Lengths: {linker_lengths}")
    print(f"Average Linker Length: {avg_linker_lengths_post}")
    print(f"Extended Linker Length: {linker_length}")

    # Plot the final linker length distribution
    plt.figure(figsize=(6,3), dpi=300)
    plt.hist(linker_lengths, bins=20, color='blue', alpha=0.7)
    plt.axvline(linker_length, color='red', linestyle='--', label='Extended')
    plt.axvline(
        avg_linker_lengths_post, color='black', linestyle='-', label='Average'
    )
    plt.xlabel('Linker Length (nm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Final Linker Length Distribution')
    plt.tight_layout()
    plt.savefig('tests/plots/final_linker_length_distribution.png', dpi=300)

    # Check that the linker lengths are close to the setpoint
    assert np.isclose(
        avg_linker_lengths_pre, avg_linker_lengths_post, atol=0.3
    ), "Chain growth algorithm and simulation are inconsistent."


if __name__ == "__main__":
    test_linker_lengths()
    print("All tests passed.")
