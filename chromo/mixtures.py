"""Mixtures represent collections of interacting polymers.
"""
# Built-in Modules
from typing import (List, Tuple)

# External Modules
import numpy as np
from sklearn.metrics import pairwise_distances

# Custom Modules
from beads import Bead
from polymers import Polymer


class Mixture:
    """Class representation of a mixture of multiple polymers.
    """

    def __init__(self, polymers: List[Polymer]):
        """Initialize the Mixture object.

        Parameters
        ----------
        polymers : List[Polymer]
            Collection of polymers forming the mixture.
        """
        self.polymers = polymers
        self.num_nucleosomes = self.count_nucleosomes()

    def count_nucleosomes(self) -> int:
        """Calculate the total number of nucleosomes in all polymers.

        Returns
        -------
        count : int
            Number of nucleosomes in all polymers
        """
        count = 0
        for poly in self.polymers:
            count += len(poly.r)
        return count

    def get_neighbors(
        self,
        radius: float
    ) -> List[Tuple[Bead, Bead]]:
        """Get neighboring beads in polymer mixture.

        NOTE: Determination of pairwise neighbors requires pairwise calculation
        of distances between all beads on all polymers. As such, this method is
        computationally intensive and recommended for use only on polymers with
        low numbers of total beads.

        Parameters
        ----------
        radius : float
            Cut-off distance used to specify a neighboring bead pair

        Returns
        -------
        List[Tuple[Bead, Bead]]
            List of neighboring bead pairs falling in specified distance
        """
        IDs = np.empty((self.num_nucleosomes, 1))
        start_inds = [0]

        for i in range(len(self.polymers)):
            poly = self.polymers[i]
            len_poly = len(poly.r)
            ind = start_inds[i]
            end_ind = ind + len_poly
            start_inds.append(end_ind)
            IDs[ind:end_ind] = i

            if i == 0:
                all_r = poly.r
            else:
                all_r = np.concatenate(all_r, poly.r)

        distances = pairwise_distances(all_r)
        nbrs = np.where(np.less_equal(distances, radius))
        nbrs = np.unique(np.sort(nbrs, axis=1), axis=0)

        neighbors = []
        for nbr in nbrs:
            poly_ID_0 = IDs[nbrs[i, 0]]
            bead_0 = self.polymers[poly_ID_0].beads[
                nbrs[i, 0] - start_inds[poly_ID_0]
            ]
            poly_ID_1 = IDs[nbrs[i, 1]]
            bead_1 = self.polymers[poly_ID_1].beads[
                nbrs[i, 1] - start_inds[poly_ID_1]
            ]
            neighbors.append((bead_0, bead_1))

        return neighbors
