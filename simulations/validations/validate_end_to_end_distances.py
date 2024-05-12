"""Validate End-to-End Distances.
"""

import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chromo.mc as mc
from chromo.polymers import SSWLC
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name
import chromo.util.poly_stat as ps

# Specify (Null) Reader Proteins
null_binder = chromo.binders.get_by_name('null_reader')
binders = chromo.binders.make_binder_collection([null_binder])

# Instantiate Polymer
num_beads = 500
bead_spacing = np.ones(num_beads-1) * 25
lp = 100

polymer = SSWLC.gaussian_walk_polymer(
    'poly_1',
    num_beads,
    bead_spacing,
    lp=lp,
    binder_names=np.array(["null_reader"])
)

# Instantiate Field
n_bins_x = 63
n_bins_y = n_bins_x
n_bins_z = n_bins_x

x_width = 1000
y_width = x_width
z_width = x_width

udf = UniformDensityField(
    polymers = [polymer],
    binders = binders,
    x_width = x_width,
    nx = n_bins_x,
    y_width = y_width,
    ny = n_bins_y,
    z_width = z_width,
    nz = n_bins_z
)

# Specify simulation
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(
    polymers = [polymer]
)
out_dir = "output_semiflex"
latest_sim = get_unique_subfolder_name(f"{out_dir}/sim_")
moves_to_use = ctrl.all_moves_except_binding_state(
    log_dir=latest_sim,
    bead_amp_bounds=amp_bead_bounds.bounds,
    move_amp_bounds=amp_move_bounds.bounds,
    controller=ctrl.SimpleControl
)
num_snapshots = 200
mc_steps_per_snapshot = 10000

# Run simulation
mc.polymer_in_field(
    polymers = [polymer],
    binders = binders,
    field = udf,
    num_save_mc = mc_steps_per_snapshot,
    num_saves = num_snapshots,
    bead_amp_bounds = amp_bead_bounds,
    move_amp_bounds = amp_move_bounds,
    output_dir = out_dir,
    mc_move_controllers = moves_to_use
)

# Evaluate Simulation
sim_dir = latest_sim

# List (sorted) snapshots
snaps = os.listdir(sim_dir)
snaps = [snap for snap in snaps if snap.startswith("poly_") and snap.endswith(".csv")]
snap_inds = [int(snap.split(".")[0].split("-")[-1])for snap in snaps]
snaps = [snap for _, snap in sorted(zip(snap_inds, snaps))]
snap_inds = np.sort(snap_inds)
print(f"Snapshots: {snaps}")

# Identify equilibrated snapshots
n_equilibrate = 150
snaps = np.array(snaps)
equilibrated_snaps = snaps[snap_inds >= n_equilibrate]

# Compute average squared end-to-end distances
n_beads = np.arange(1, 100)
seg_lengths = n_beads * bead_spacing[0] / (2 * lp)
e2e = []
for snap in equilibrated_snaps:
    e2e_snap = []
    snap_path = os.path.join(sim_dir, snap)
    r = pd.read_csv(snap_path, sep=",", header=[0, 1], index_col=0)["r"].to_numpy()
    for n_beads_ in n_beads:
        r1 = r[n_beads_:]
        r2 = r[:-n_beads_]
        e2e_snap.append(
            np.average(
                np.linalg.norm(r1 - r2, axis=1) ** 2
            ) / ((2 * lp)**2)
        )
    e2e.append(e2e_snap)
e2e = np.array(e2e)
e2e = np.average(e2e, axis=0)
assert len(e2e) == len(seg_lengths), "End-to-end distances do not align with segment lengths."

# Compute theoretical end-to-end distances
r2_theory = seg_lengths - 1/2 + np.exp(-2 * seg_lengths)/2

# Plot mean squared end-to-end distances
plt.figure(figsize=(4,3), dpi=300)
plt.scatter(seg_lengths, e2e, color="black", label="simulation")
plt.xlabel(r"$L/(2l_p)$")
plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
plt.plot(seg_lengths, r2_theory, color="red", label="theory")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(sim_dir, "e2e_distances.png"), dpi=600)
