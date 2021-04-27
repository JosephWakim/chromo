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

"""
# Evaluate functionality of individual moves
move_amps = [0.05, 0.25, 1, 5, 10]
bead_amps = [5, 10, 25, 50, 100]

# Evaluate the Crank-shaft move
print("Crank-shaft move")
for move_amp in move_amps:
    for bead_amp in bead_amps:
        p = Polymer.straight_line_in_x(
            'Chr-1',
            num_beads,
            0.2,
            states=np.zeros((num_beads,)),
            mark_names=['HP1']
        )
        print("Move amp: ", str(move_amp))
        print("Bead amp: ", str(bead_amp))
        move = mc.moves.MCAdapter(mc.moves.crank_shaft)
        move.amp_move = move_amp
        move.amp_bead = bead_amp
        mc.polymer_in_field(
            [p],
            marks,
            udf,
            1,
            100,
            output_dir='output/testing_individual_moves/crank_shaft', 
            mc_moves=[move]
        )

# Evaluate the End-pivot move
print("End-pivot move")
for move_amp in move_amps:
    for bead_amp in bead_amps:
        p = Polymer.straight_line_in_x(
            'Chr-1',
            num_beads,
            0.2,
            states=np.zeros((num_beads,)),
            mark_names=['HP1']
        )
        print("Move amp: ", str(move_amp))
        print("Bead amp: ", str(bead_amp))
        move = mc.moves.MCAdapter(mc.moves.end_pivot)
        move.amp_move = move_amp
        move.amp_bead = bead_amp
        mc.polymer_in_field(
            [p],
            marks,
            udf,
            1,
            100,
            output_dir='output/testing_individual_moves/end_pivot',
            mc_moves=[move]
        )

# Evaluate the Slide move
print("Slide move")
for move_amp in move_amps:
    for bead_amp in bead_amps:
        p = Polymer.straight_line_in_x(
            'Chr-1',
            num_beads,
            0.2,
            states=np.zeros((num_beads,)),
            mark_names=['HP1']
        )
        print("Move amp: ", str(move_amp))
        print("Bead amp: ", str(bead_amp))
        move = mc.moves.MCAdapter(mc.moves.slide)
        move.amp_move = move_amp
        move.amp_bead = bead_amp
        mc.polymer_in_field(
            [p],
            marks,
            udf,
            1,
            100,
            output_dir='output/testing_individual_moves/slide',
            mc_moves=[move]
        )

# Evaluate the Tangent-rotation move
print("Tangent-rotation move")
for move_amp in move_amps:
    for bead_amp in bead_amps:
        p = Polymer.straight_line_in_x(
            'Chr-1',
            num_beads,
            0.2,
            states=np.zeros((num_beads,)),
            mark_names=['HP1']
        )
        print("Move amp: ", str(move_amp))
        print("Bead amp: ", str(bead_amp))
        move = mc.moves.MCAdapter(mc.moves.tangent_rotation)
        move.amp_move = move_amp
        move.amp_bead = bead_amp
        mc.polymer_in_field(
            [p],
            marks,
            udf,
            1,
            100,
            output_dir='output/testing_individual_moves/tangent_rotation',
            mc_moves=[move]
        )

# Reinitialize polymer
p = Polymer.straight_line_in_x(
    'Chr-1',
    num_beads,
    0.2,
    states=np.zeros((num_beads,)),
    mark_names=['HP1']
)
"""

# Evaluate performance of the simulator for all moves
mc.polymer_in_field([p], marks, udf, 1000, 100000, output_dir='output')
