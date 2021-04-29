"""Run a basic simulation from a script.

This simulation runs a Monte Carlo simulation from a script for easier use of
a remote tower.
"""

# Built-in Modules
import importlib
import os
import sys
from io import StringIO

# External Modules
import numpy as np
import pandas as pd

# Change working directory
cwd = os.getcwd()
os.chdir("../..")
print("Current Working Directory: ")
print(os.getcwd())
print("System path: ")
print(sys.path)

# Custom Modules
import chromo
import chromo.mc as mc
from chromo.components import Polymer
import chromo.marks
from chromo.fields import UniformDensityField

# Specify epigenetic mark
epimarks = [chromo.marks.get_by_name('HP1')]
print("Epigenetic marks: ")
print(epimarks)

# Reformat epigenic marks into a dataframe format
marks = chromo.marks.make_mark_collection(epimarks)

# Specify polymers length
num_beads = 10000
p = Polymer.straight_line_in_x(
    'Chr-1',
    num_beads,
    0.2,
    states=np.zeros((num_beads,)),
    mark_names=['HP1']
)

# Specify the field containing the polymers
udf = UniformDensityField([p], marks, 20, 20, 20, 20, 20, 20)

# Evaluate performance of the simulator for all moves
mc.polymer_in_field([p], marks, udf, 1000, 1000, output_dir='output')
