"""Continue a Monte Carlo simulation from a savepoint.

Joseph Wakim
June 30, 2021
"""

# Built-in Modules
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

# Custom Modules
import chromo.mc as mc
from chromo.polymers import (Chromatin)
import chromo.marks
from doc.source.mc import marks, udf

polymer_class = Chromatin
output_dir = "output"
num_save_mc = 500
num_saves = 2

mc.continue_polymer_in_field_simulation(
    polymer_class,
    marks,
    udf,
    output_dir,
    num_save_mc,
    num_saves
)
