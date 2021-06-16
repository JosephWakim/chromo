"""Evaluate polymer statistics from a script.
"""

import os
import sys
from typing import Optional

import matplotlib.pyplot as plt

from doc.source.poly_vis import get_latest_simulation
import chromo.util.poly_stat as ps


def main(sim_ID: Optional[int] = None, window: Optional[int] = 10):
    """Calculate average end-to-end distance for a given bead window.

    Parameters
    ----------
    sim_ID : Optional[int]
        Serial identifier for simulation being visualized (default = None)
    window : Optional[int]
        Number of monomer units for which to calculate average squared E2E
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cwd + '/../../output')

    if sim_ID is None:
        sim = get_latest_simulation()
    else:
        sim = "sim_" + str(sim_ID)

    print("Sim: " + sim)
    output_dir = os.getcwd() + '/' + sim
    output_files = os.listdir(output_dir)
    output_files = [
        f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
    ]
    snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
    output_files = [f for _, f in sorted(zip(snapshot, output_files))]

    os.chdir(cwd)

    r2 = []
    for i, f in enumerate(output_files):
        if (i+1) % 10 == 0:
            print("Snapshot: " + str(i+1) + " of " + str(len(output_files)))
            print("File: " + f)
            print()

        poly_stat = ps.PolyStat("output/" + sim + "/" + f)
        r2.append(
            poly_stat.get_avg_r2(
                sampling_scheme="overlap_slide", bead_separation=window
            )
        )

    os.chdir(output_dir)
    plt.plot(r2)
    plt.xlabel("Snapshot")
    plt.ylabel("Mean R2 (10 nucleosomes)")
    plt.savefig("Mean_E2E_10_Nucleosomes.png", dpi=600)
    os.chdir(cwd)


if __name__ == "__main__":
    """Evaluate Average squared End-to-End distance from simulations.
    """
    args = sys.argv

    if len(args) == 1:
        main()
    elif len(args) == 2:
        try:
            ID = int(args[1])
        except ValueError:
            print("Enter simulation numbers for E2E evaluation as integer.")
            sys.exit()
        main(ID)
    else:
        try:
            ID = int(args[1])
            window = int(args[2])
        except ValueError:
            print(
                "Enter simulation numbers and window size for E2E evaluation \
                as integers."
            )
            sys.exit()
        main(ID, window)
