"""Run a basic simulation from a script.

This simulation runs a Monte Carlo simulation from a script for easier use of
a remote tower.
"""

# Built-in Modules
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

# External Modules
import numpy as np

# Custom Modules
import chromo
import chromo.mc as mc
from chromo.polymers import (
    Chromatin, helix_parametric_x, helix_parametric_y, helix_parametric_z
)
import chromo.marks
from chromo.fields import UniformDensityField

# Change working directory
os.chdir("../..")
print("Current Working Directory: ")
print(os.getcwd())
print("System path: ")
print(sys.path)

# Specify epigenetic mark
epimarks = [chromo.marks.get_by_name('HP1')]
print("Epigenetic marks: ")
print(epimarks)

# Reformat epigenic marks into a dataframe format
marks = chromo.marks.make_mark_collection(
    epimarks
)

print("Constructing polymer...")
# Specify polymers length
num_beads = 1000
bead_spacing = 7.8     # 1/20 persistence length
p = Chromatin.arbitrary_path_in_x_y_z(
    'Chr-1',
    num_beads,
    bead_spacing,
    shape_func_x=helix_parametric_x,
    shape_func_y=helix_parametric_y,
    shape_func_z=helix_parametric_z,
    states=np.zeros((num_beads,)),
    mark_names=['HP1']
)

# Specify the field containing the polymers
x_width = 20
n_bins_x = 20
y_width = 20
n_bins_y = 20
z_width = 20
n_bins_z = 20
udf = UniformDensityField(
    [p], marks, x_width, n_bins_x, y_width,
    n_bins_y, z_width, n_bins_z
)

# Specify the bead selection and move amplitude bounds
polymers = [p]
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers)

if __name__ == "__main__":
    """Run the simulation.
    """
    print("Starting new simulation...")
    num_snapshots = 200
    mc_steps_per_snapshot = 500
    mc.polymer_in_field(
        [p],
        marks,
        udf,
        mc_steps_per_snapshot,
        num_snapshots,
        amp_bead_bounds,
        amp_move_bounds,
        output_dir='output'
    )
