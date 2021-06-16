"""Visualize polymer state for all simulations.

Usage: `python poly_vis.py ## ## ## ## ...`
where ## represents a serial index for a simulation to be visualized
"""

# Built-in Modules
import os
import sys
from typing import Optional

# External Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def get_latest_simulation():
    """Get the latest simulation file from the cwd.

    Return
    ------
    str
        Most recent simulation output directory.
    """
    all_items = os.listdir()
    simulations = [item for item in all_items if os.path.isdir(item)]
    sim_IDs = [int(sim.split("_")[-1]) for sim in simulations]
    latest_sim_ID = np.argmax(sim_IDs)
    return simulations[latest_sim_ID]


def main(sim_ID: Optional[int] = None):
    """Generate images of polymer configurations.

    Images of the polymer at each snapshot will be saved to the output
    directory corresponding to the serial simulation ID specified in the
    command line.

    These visualizations are not meant to be publication quality, but are
    intended to help spot-check and debug.

    Parameters
    ----------
    sim_ID : Optional[int]
        Serial identifier for simulation being visualized (default = None)
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cwd + '/../../output')

    if sim_ID is None:
        sim = get_latest_simulation()
    else:
        sim = "sim_" + str(sim_ID)

    print("Sim: " + sim)

    sim_dir = os.getcwd() + "/" + sim
    all_items = os.listdir(sim_dir)
    snapshots = [
        item for item in all_items if
        item.endswith(".csv") and item.startswith("Chr")
    ]
    snapshots.sort()

    for snap in snapshots:

        data_path = sim_dir + "/" + snap
        snap_data = pd.read_csv(
            data_path,
            skiprows=1,
            index_col=0,
            usecols=[0, 1, 2, 3]
        )
        x = snap_data.x.values
        y = snap_data.y.values
        z = snap_data.z.values

        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlabel('z', fontsize=16)

        plt.savefig(sim_dir+"/vis_"+snap.split(".")[0]+".png")
        plt.close()

    os.chdir(cwd)


if __name__ == "__main__":
    """Generate visuals.
    """
    args = sys.argv

    if len(args) == 1:
        main()
    else:
        for i in range(1, len(args)):
            try:
                ID = int(args[i])
            except ValueError:
                print(
                    "Enter simulation numbers for visualization as integers."
                )
                sys.exit()
            main(ID)
