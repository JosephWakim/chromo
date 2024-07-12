"""Simulate a chromatin fiber with steric interactions.
"""

import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

import numpy as np
from inspect import getmembers, isfunction

import chromo.mc as mc
import chromo.polymers as poly
from chromo.polymers import Chromatin
import chromo.binders
from chromo.fields import NullField
import chromo.util.mu_schedules as ms
from wlcstat.chromo import gen_chromo_conf

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())

# Define linker lengths
rev_sim_dir = "/home/users/jwakim/chromo_variable_linkers/simulations/examples"
linker_length_file = os.path.join(
    rev_sim_dir, "example_links_0p09_for_revisions.csv"
)
linker_lengths_bp = np.loadtxt(linker_length_file)
length_bp = 0.332
bp_wrap = 147.
lp = 50
lt = 100

# Load chemical modifications
chem_mod_file = os.path.join(
    rev_sim_dir, "example_methylation_0p09_for_revisions.csv"
)
chemical_mod_path = np.array([chem_mod_file])
chemical_mods = Chromatin.load_seqs(chemical_mod_path)
modification_name = "H3K9me3"

# Correct the linker lengths
# The theory includes an extra linker to account for the first nucleosome.
# The simulation does not include this extra linker.
linker_lengths_bp = linker_lengths_bp[:-1]
linker_lengths = linker_lengths_bp * length_bp
n_beads = len(linker_lengths) + 1

# Instantiate the HP1 reader protein
binder = chromo.binders.get_by_name('HP1')
binder.chemical_potential = float(sys.argv[1])

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
num_snapshots = 200
mc_steps_per_snapshot = 200

# Create a list of mu schedules, which are defined in another file
schedules = [func[0] for func in getmembers(ms, isfunction)]
select_schedule = "linear_step_for_negative_cp"
mu_schedules = [
    ms.Schedule(getattr(ms, func_name)) for func_name in schedules
]
mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]

random_seed = np.random.randint(1, 100000)
output_dir = "output_6"

path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"

p_sim = mc.polymer_in_field(
    [p],
    binders,
    field,
    mc_steps_per_snapshot,
    num_snapshots,
    amp_bead_bounds,
    amp_move_bounds,
    output_dir=output_dir,
    mu_schedule=mu_schedules[0],
    random_seed=random_seed,
    path_to_run_script=path_to_run_script,
    path_to_chem_mods=chemical_mod_path,
    run_command=run_command
)
