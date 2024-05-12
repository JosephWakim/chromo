"""Validate radial density distribution of a flexible homopolymer.
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

# Specify Confinement
confine_type = "Spherical"
confine_length = 1000

# Instantiate Polymer
num_beads = 10000
bead_spacing = 100
lp = 0.5

polymer = SSWLC.confined_gaussian_walk(
    'poly_1',
    num_beads,
    bead_spacing,
    confine_type=confine_type,
    confine_length=confine_length,
    binder_names=np.array(['null_reader']),
    lp=lp
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
    nz = n_bins_z,
    confine_type = confine_type,
    confine_length = confine_length,
    chi=0,
    vf_limit=1.0
)

# Specify Simulation
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(
    polymers = [polymer]
)
output_dir = "output_flex_confined"
latest_sim = get_unique_subfolder_name(f"{output_dir}/sim_")
moves_to_use = ctrl.all_moves_except_binding_state(
    log_dir=latest_sim,
    bead_amp_bounds=amp_bead_bounds.bounds,
    move_amp_bounds=amp_move_bounds.bounds,
    controller=ctrl.SimpleControl
)

num_snapshots = 200
mc_steps_per_snapshot = 4000

# Run Simulation
mc.polymer_in_field(
    polymers = [polymer],
    binders = binders,
    field = udf,
    num_save_mc = mc_steps_per_snapshot,
    num_saves = num_snapshots,
    bead_amp_bounds = amp_bead_bounds,
    move_amp_bounds = amp_move_bounds,
    output_dir = output_dir,
    mc_move_controllers = moves_to_use
)

# Evaluate Radial Density Distribution
sim_dir = os.path.join(parent_dir, output_dir, latest_sim)

# List (sorted) snapshots
snaps = os.listdir(sim_dir)
snaps = [snap for snap in snaps if snap.startswith("poly_") and snap.endswith(".csv")]
snap_inds = [int(snap.split(".")[0].split("-")[-1])for snap in snaps]
snaps = [snap for _, snap in sorted(zip(snap_inds, snaps))]
snap_inds = np.sort(snap_inds)

# Identify equilibrated snapshots
n_equilibrate = 175
snaps = np.array(snaps)
equilibrated_snaps = snaps[snap_inds >= n_equilibrate]
step_size = 50

# Compute Radial Densities
counts_all = []
for snap in equilibrated_snaps:
    snap_path = os.path.join(sim_dir, snap)
    r = pd.read_csv(snap_path, sep=",", header=[0, 1], index_col=0)["r"].to_numpy()
    radial_dists = np.linalg.norm(r, axis=1)
    bins = np.arange(step_size, confine_length, step_size)
    counts, bin_edges = np.histogram(radial_dists, bins=bins)
    counts = counts.astype(float)
    counts_all.append(counts)

counts_all = np.array(counts_all)
counts_avg = np.sum(counts_all, axis=0)

# Correct densities based on volumes of spherical shells
for i in range(len(bin_edges)-1):
    volume = 4/3 * np.pi * ((bin_edges[i+1]/1E3)**3 - (bin_edges[i]/1E3)**3)
    counts_avg[i] /= volume
counts_avg /= np.sum(counts_avg)

# Get theoretical radial densities
a = confine_length
b = lp
N = len(r)
r_theory = np.arange(step_size, confine_length, 1)
n_max = 1000
rho = np.zeros(len(r_theory))
for n in range(2, n_max + 1):
    rho += (-1)**(n+1) / (n * np.pi) * np.sin(np.pi * r_theory / a) * np.sin(n * np.pi * r_theory / a) / (r_theory**2 * b**2 * (n**2 - 1))
rho += N / np.pi * np.sin(np.pi * r_theory / a)**2 / r_theory**2

normalize = np.sum(rho)
rho_theory = rho / normalize * step_size

plt.figure(figsize=(4,3), dpi=300)
plt.hist(bin_edges[:-1], bin_edges, weights=counts_avg, alpha=1, color="gray", label="simulation")
plt.plot(r_theory, rho_theory, color="red", label="theory")
plt.xlabel("Radial Distance (nm)")
plt.ylabel(r"Probability")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(sim_dir, "radial_density.png"), dpi=600)
