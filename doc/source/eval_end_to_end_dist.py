
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from doc.source.poly_vis import get_latest_simulation
import chromo.util.poly_stat as ps
import chromo.polymers as polymers

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

sim = get_latest_simulation()
# sim = "sim_20"
num_equilibration_steps = 90

print("Sim: " + sim)
output_dir = os.getcwd() + '/' + sim
output_files = os.listdir(output_dir)
output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
]
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
sorted_snap = np.sort(np.array(snapshot))
output_files = [f for _, f in sorted(zip(snapshot, output_files))]
output_files = [
    output_files[i] for i in range(len(output_files))
    if sorted_snap[i] > num_equilibration_steps - 1
]

os.chdir(parent_dir)

log_vals = np.arange(-2, 3, 0.05)
bead_range = 10 ** log_vals * 80
bead_range = bead_range.astype(int)
bead_range = np.array(
    [bead_range[i] for i in range(len(bead_range)) if bead_range[i] > 0]
)
bead_range = np.unique(bead_range)
average_squared_e2e = np.zeros((len(bead_range), 1))

for j, window_size in enumerate(bead_range):
    if window_size > 0:
        print("!!!!! WINDOW SIZE: " + str(window_size) + " !!!!!")
        r2 = []
        for i, f in enumerate(output_files):
            if (i+1) % 10 == 0:
                print("Snapshot: " + str(i+1) + " of " + str(len(output_files)))
                print()

            output_path = "output/" + sim + "/" + f
            polymer = polymers.Chromatin.from_file(output_path, name=f)
            # Retrieve the kuhn length only once, since it is the same
            if j == 0 and i == 0:
                kuhn_length = 2 * polymer.lp
            poly_stat = ps.PolyStats(polymer, "overlap")
            r2.append(
                poly_stat.calc_r2(
                    windows=poly_stat.load_indices(window_size)
                )
            )
        average_squared_e2e[j] = np.average(r2)

bead_range = bead_range / 80     # Convert x axis to number of kuhn lengths

os.chdir(output_dir)
with open("avg_squared_e2e.txt", "w") as output_file:
    for val in average_squared_e2e:
        output_file.write('%s\n' % val)

plt.figure()
plt.scatter(bead_range, average_squared_e2e)
plt.xlabel(r"$L/(2l_p)$")
plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
plt.yscale("log")
plt.xscale("log")
plt.savefig("Squared_e2e_vs_dist_v2.png", dpi=600)
os.chdir(cwd)

# Plot the mean squared end-to-end distance on a log-log plot
os.chdir(output_dir)
plt.figure()
plt.scatter(np.log10(bead_range), np.log10(average_squared_e2e))
plt.xlabel(r"Log $L/(2l_p)$")
plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
r2_theory = 2 * (bead_range / 2 - (1 - np.exp(-(2) * bead_range)) / (2) ** 2)
plt.plot(np.log10(bead_range), np.log10(r2_theory))
plt.savefig("Log_Log_Squared_e2e_vs_dist_v2.png", dpi=600)
os.chdir(cwd)
