"""Compare simulated and theoretical end-to-end distances of a kinked twistable wlc.
"""

import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

import numpy as np
import chromo.mc as mc
import chromo.polymers as poly
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())

# Define chromatin to match theory
linker_length_bp = 36
length_bp = 0.332
linker_length = linker_length_bp * length_bp
n_beads = 50
linker_lengths = np.array([linker_length] * (n_beads-1))
lp = 10000.
lt = 10000.
bp_wrap = 147.

# Instantiate the HP1 reader protein
# This is pre-defined in the `chromo.binders` module
null_binder = chromo.binders.get_by_name('null_reader')

# Create the binder collection
binders = chromo.binders.make_binder_collection([null_binder])

# Initialize the polymer
p = poly.DetailedChromatin2.straight_line_in_x(
    "Chr",
    linker_lengths,
    bp_wrap=bp_wrap,
    lp=lp,
    lt=lt,
    binder_names=np.array(["null_reader"])
)

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
    nz=n_bins_z
)

amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers=[p])

latest_sim = get_unique_subfolder_name("output/sim_")
moves_to_use = ctrl.all_moves_except_binding_state(
    log_dir=latest_sim,
    bead_amp_bounds=amp_bead_bounds.bounds,
    move_amp_bounds=amp_move_bounds.bounds,
    controller=ctrl.SimpleControl
)

num_snapshots = 200
mc_steps_per_snapshot = 2000

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
    output_dir='output',
    random_seed=random_seed,
    mc_move_controllers=moves_to_use
)
