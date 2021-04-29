"""Evaluate polymer statistics from a script.
"""

import importlib
import os
import sys
from io import StringIO

import numpy as np
import pandas as pd

# Adjust current working directory
cwd = os.getcwd()
os.chdir("../..")
print("Current Working Directory: ")
print(os.getcwd())
print()

# Load polymer statistics module
import chromo.util.poly_stat as ps

# Specify simulation ID on which to run the analysis
sim_ID = 3

# Open the output directory
os.chdir(cwd + "/../../")
output_dir = os.getcwd() + "/output/sim_" + str(sim_ID)

# Generate a list of CSV configuration files in the output directory
output_files = os.listdir(output_dir)
output_files = [f for f in output_files if f.endswith(".csv")]
output_files = [f for f in output_files if len(f.split("_")) == 1]

# Sort the CSV configuration files by snapshot number
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
output_files = [f for _, f in sorted(zip(snapshot, output_files))]

# Calculate end-to-end distances
os.chdir(cwd + "/../../chromo/util")

r2 = []
for i, f in enumerate(output_files):
    if (i+1) % 10 == 0:
        print("Snapshot: " + str(i+1) + " of " + str(len(output_files)))
        print("File: " + f)
        print()
    
    poly_stat = ps.PolyStat("output/sim_" + str(sim_ID) + "/" + f)
    r2.append(
        poly_stat.get_avg_r2(sampling_scheme="overlap_slide", bead_separation=10)
    )

# Visualize Polymer Statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.chdir(output_dir)
plt.plot(r2)
plt.xlabel("Snapshot")
plt.ylabel("Mean R2 (10 nucleosomes)")
plt.savefig("Mean_E2E_10_Nucleosomes.png", dpi=600)
