"""Run the refinement step of a simulation with sequential coarse-graining.

By:     Joseph Wakim
Group:  Spakowitz Lab @ Stanford
Date:   June 10, 2024

Usage:  python two_mark_refine.py CONFIG_PATH
where:  CONFIG_PATH provides the path to a JSON configuration file that
        includes the following key-value pairs:

            1. CG_output: path to the output directory for the SINGLE
                coarse-grained simulation
            2. CG_config_file_name : name of the configuration file in the
                coarse-grained output directory
            3. modification_sequence_path_1 : path to a file containing the
                full-resolution modification sequence for the first mark
            4. modification_sequence_path_2 : path to a file containing the
                full-resolution modification sequence for the second mark
            5. output_dir (optional) : Path to the new output directory for ALL
                refined simulations. If the output directory does not exist, it
                will be created. The output directory will be set to
                "output_refined" if not specified in the configuration file.
            6. random_seed (optional) : If specified, a random seed will be
                set accordingly. If not specified, a new random seed will be
                generated.
            7. refine_ratio (optional) : Number of refined beads per
                coarse-grained bead. If not specified, a default value of 10
                will be used, indicating that each coarse grained bead
                represents 10 refined beads.

Notes:  There is no simulated annealing performed during the refinement step.
        As currently implemented, the refinement step only handles uniform
        linker lengths.
"""

# Import Modules
import os
import sys
from inspect import getmembers, isfunction
import json

import numpy as np
import pandas as pd

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)
os.chdir(parent_dir)

print("Root Directory: ")
print(os.getcwd())

import chromo.mc as mc
from chromo.polymers import Chromatin
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.util.rediscretize as rd
import chromo.util.mu_schedules as ms

# Load simulation parameters from config file
config_file_path = sys.argv[1]
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)
required_keys = [
    "CG_output",
    "CG_config_file_name",
    "modification_sequence_path_1",
    "modification_sequence_path_2",
]
for key in required_keys:
    assert key in config.keys(), f"Missing required key: {key}"

# Set the random seed
if "random_seed" in config:
    random_seed = config["random_seed"]
else:
    random_seed = np.random.randint(0, 1E5)
np.random.seed(random_seed)

# Load the refinement ratio
if "refine_ratio" in config:
    refine_ratio = config["refine_ratio"]
else:
    refine_ratio = 10

# Load coarse-grained simulation
cg_dir = config["CG_output"]
binder_path = os.path.join(cg_dir, "binders")
udf_path = os.path.join(cg_dir, "UniformDensityField")

# Load the latest snapshot from the coarse-grained simulation
files = os.listdir(cg_dir)
polymer_prefix = "Chr"
files = [
    file for file in files
    if file.endswith(".csv") and file.startswith(polymer_prefix)
]
snaps = [int(file.split(".")[0].split("-")[-1]) for file in files]
files = [file for _, file in sorted(zip(snaps, files))]
latest_snap = files[-1]
latest_snap_path = os.path.join(cg_dir, latest_snap)

# Create a coarse-grained polymer
p_cg = Chromatin.from_file(latest_snap_path, name="Chr_CG")
num_beads_cg = p_cg.num_beads
num_beads = int(num_beads_cg * refine_ratio)

# Bead density is defined by MacPherson et al. 2018
bead_density = 393216 / (4 / 3 * np.pi * 900 ** 3)

# Specify confinement
confine_type = "Spherical"
confine_length = (num_beads / bead_density / (4 / 3 * np.pi)) ** (1 / 3)

# Specify chemical modifications
chem_mods_path = np.array([
    config["modification_sequence_path_1"],
    config["modification_sequence_path_2"]
])
chemical_mods = Chromatin.load_seqs(chem_mods_path)[:num_beads]

# Load the coarse-grained simulation parameters
cg_config_path = os.path.join(
    config["CG_output"], config["CG_config_file_name"]
)
with open(cg_config_path, "r") as cg_config_file:
    config_cg = json.load(cg_config_file)

# Load binder properties
# Two mark simulations are hard coded for binders HP1 and PRC1
# Properties of these marks can be adjusted
binder_1 = chromo.binders.get_by_name("HP1")
binder_2 = chromo.binders.get_by_name("PRC1")
binders = [binder_1, binder_2]

# Update the binder parameters
binders[0].chemical_potential = config_cg["chemical_potential_1"]
binders[1].chemical_potential = config_cg["chemical_potential_2"]
binders[0].interaction_energy = config_cg["self_interaction_energy_1"]
binders[1].interaction_energy = config_cg["self_interaction_energy_2"]
binders[0].cross_talk_interaction_energy["PRC1"] = \
    config_cg["cross_interaction_energy"]

# Create the binder collection
binders = chromo.binders.make_binder_collection(binders)

# Load field parameters from coarse-grained simulation
field_params = pd.read_csv(
    udf_path, header=None, names=["Attribute", "Value"], index_col=0
)
x_width = float(field_params.loc["x_width", "Value"])
y_width = float(field_params.loc["y_width", "Value"])
z_width = float(field_params.loc["z_width", "Value"])
nx = int(field_params.loc["nx", "Value"])
ny = int(field_params.loc["ny", "Value"])
nz = int(field_params.loc["nz", "Value"])
confine_type_cg = field_params.loc["confine_type", "Value"]
confine_length_cg = float(field_params.loc["confine_length", "Value"])
chi = float(field_params.loc["chi", "Value"])
assume_fully_accessible = (
        field_params.loc["assume_fully_accessible", "Value"] == "True"
)
fast_field = int(field_params.loc["fast_field", "Value"] == "True")

# Create coarse-grained field
udf_cg = UniformDensityField(
    [p_cg], binders, x_width, nx, y_width, ny, z_width, nz,
    confine_type=confine_type_cg, confine_length=confine_length_cg,
    chi=chi, assume_fully_accessible=assume_fully_accessible,
    fast_field=fast_field
)

# Store details on simulation
path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"
root_dir = "/".join(path_to_run_script.split("/")[:-3])
if "output_dir" in config:
    output_dir = config["output_dir"]
else:
    output_dir = os.path.join(root_dir, "output_refined")

# Refine the polymer
n_bind_eq = 1000000
bead_spacing = 16.5  # Currently, only uniform linker lengths are supported
p_refine, udf_refine = rd.refine_chromatin(
    polymer_cg=p_cg,
    num_beads_refined=num_beads,
    bead_spacing=bead_spacing,
    chemical_mods=chemical_mods,
    udf_cg=udf_cg,
    binding_equilibration=n_bind_eq,
    name_refine="Chr_refine",
    output_dir=output_dir
)

# Specify move and bead amplitudes
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds([p_refine])

# Specify duration of simulation
num_snapshots = 200
mc_steps_per_snapshot = 5000

# Run the refined simulation
polymers_refined = mc.polymer_in_field(
    [p_refine],
    binders,
    udf_refine,
    mc_steps_per_snapshot,
    num_snapshots,
    amp_bead_bounds,
    amp_move_bounds,
    output_dir=output_dir,
    random_seed=random_seed,
    path_to_run_script=path_to_run_script,
    path_to_chem_mods=chem_mods_path,
    run_command=run_command
)
