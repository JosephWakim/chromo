
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from doc.source.poly_vis import get_latest_simulation
import chromo.util.poly_stat as ps

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

sim = get_latest_simulation()
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

os.chdir(cwd)

bead_range = np.arange(1, 400, 1)
average_squared_e2e = np.zeros((len(bead_range), 1))

for j, window_size in enumerate(bead_range):
    print("!!!!! WINDOW SIZE: " + str(window_size) + " !!!!!")
    r2 = []
    for i, f in enumerate(output_files):
        if (i+1) % 10 == 0:
            print("Snapshot: " + str(i+1) + " of " + str(len(output_files)))
            print()

        poly_stat = ps.PolyStat("output/" + sim + "/" + f)
        r2.append(
            poly_stat.get_avg_r2(
                sampling_scheme="overlap_slide", bead_separation=window_size
            )
        )
    average_squared_e2e[j] = np.average(r2)

bead_range = bead_range / 50     # Convert x axis to number of kuhn lengths

os.chdir(output_dir)
with open("avg_squared_e2e.txt", "w") as output_file:
    for val in average_squared_e2e:
        output_file.write('%s\n' % val)

plt.figure()
plt.scatter(bead_range, average_squared_e2e)
plt.xlabel("Kuhn lengths")
plt.ylabel("Average squared end-to-end dist.")
plt.yscale("log")
plt.xscale("log")
plt.savefig("Squared_e2e_vs_dist_v2.png", dpi=600)
os.chdir(cwd)

os.chdir(output_dir)
plt.figure()
plt.scatter(np.log10(bead_range), np.log10(average_squared_e2e))
plt.xlabel("Log Kuhn lengths")
plt.ylabel("Log average squared end-to-end dist.")
# short_range_behavior
ref_x = [np.log10(bead_range[0])]
ref_y = [np.log10(average_squared_e2e[0])]
step = 0.01
for i in range(1, 200):
    ref_x.append(ref_x[i-1] + step)
    ref_y.append(ref_y[i-1] + 2 * step)
plt.plot(ref_x, ref_y)

ref_x = [np.log10(bead_range[-1])]
ref_y = [np.log10(average_squared_e2e[-1])]
step = 0.01
for i in range(1, 200):
    ref_x.append(ref_x[i-1] - step)
    ref_y.append(ref_y[i-1] - 1 * step)
plt.plot(ref_x, ref_y)
plt.savefig("Log_Log_Squared_e2e_vs_dist_v2.png", dpi=600)
os.chdir(cwd)
