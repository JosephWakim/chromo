"""Control MC simulation parameters.
"""

# Built-in Modules
from abc import ABC, abstractmethod
from typing import List

# External Modules
import numpy as np

# Custom modules
import chromo.mc.mc_stat as mc_stat


class Controller(ABC):
    """Class representation of controller for MC simulation parameters.
    """

    def __init__(
        self,
        init_move_amp: float,
        init_bead_amp: int,
        performance_tracker: mc_stat.PerformanceTracker
    ):
        """Initialize the MC controller.

        Parameters
        ----------
        init_move_amp : float
            Current move amplitude upon initializing controller.
        init_bead_amp : int
            Current bead selection amplitude upon initializing controller.
        performance_tracker : mc_stat.PerformanceTracker
            Object used to log move/bead amplitudes and move acceptance rates
        """
        self.move_amp = init_move_amp
        self.bead_amp = init_bead_amp
        self.performance_tracker = performance_tracker

    @abstractmethod
    def update_move_amplitude(self):
        """Update the move amplitude using `self.performance_tracker`.
        """
        pass

    @abstractmethod
    def update_bead_amplitude(self):
        """Update bead selection amplitude using `self.performance_tracker`.
        """
        pass


class NoControl(Controller):
    """Class representation of a trivial, non-acting MC controller.
    """

    def update_move_amplitude(self):
        """Trivially maintain current move amplitude.
        """
        return
    
    def update_bead_amplitude(self):
        """Trivially maintain current bead selection amplitude.
        """
        return


class SimpleControl(Controller):