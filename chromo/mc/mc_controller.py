"""Control MC simulation parameters.
"""

# Built-in Modules
from abc import ABC, abstractmethod
from typing import List, Optional

# External Modules
import numpy as np

# Custom modules
import chromo.mc.mc_stat as mc_stat
import chromo.mc.moves as mv


class Controller(ABC):
    """Class representation of controller for MC simulation parameters.
    """

    def __init__(
        self,
        mc_adapter: mv.MCAdapter
    ):
        """Initialize the MC controller.

        Parameters
        ----------
        mc_adapter: mv.MCAdapter
        """
        self.move = mc_adapter
        
    def update_amplitudes(self):
        """Update bead selection and move amplitudes.
        """
        self.update_bead_amplitude()
        self.update_move_amplitude()

    @abstractmethod
    def update_move_amplitude(self):
        """Update the move amplitude.

        Use acceptance rate `self.move.performance_tracker.acceptance_rate`
        to update the move amplitude.
        """
        pass

    @abstractmethod
    def update_bead_amplitude(self):
        """Update the bead selection amplitude.

        Use acceptance rate `self.move.performance_tracker.acceptance_rate`
        to update the move amplitude.
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
    """Apply factor changes to move or bead amplitude based on acceptance rate.

    Based on threshold cutoffs in move acceptance rate, a factor change is
    applied to the move or bead amplitudes. The change drives the acceptance
    rate towards a setpoint value of 0.5.

    Changes to the move and bead amplitudes are made based on the following
    acceptance rate thresholds:

    0 to 0.3        Multiply move amplitude by a factor between 0 and 1
    0.3 to 0.45     Multiply bead amplitude by a factor between 0 and 1
    0.45 to 0.55    No change to move or bead acceptance rate
    0.55 to 0.7     Divide bead amplitude by a factor between 0 and 1
    0.7 to 1        Divide move amplitude by a factor between 0 and 1
    """

    def update_move_amplitude(self, factor: Optional[float] = 0.95):
        """Update the move amplitude based on the acceptance rate.

        Parameters
        ----------
        factor : Optional[float]
            Factor by which to multiply or divide move/bead amplitudes in
            response to current acceptance rate (default = 0.95)
        """
        acceptance = self.move.performance_tracker.acceptance_rate
        
        if acceptance >= 0 and acceptance <= 0.3:
            self.move.amp_move *= factor
        elif acceptance >= 0.7 and acceptance <= 1:
            self.move.amp_move /= factor

    def update_bead_amplitude(self, factor: Optional[float] = 0.95):
        """Update the bead amplitude based on the acceptance rate.

        Parameters
        ----------
        factor : Optional[float]
            Factor by which to multiply or divide move/bead amplitudes in
            response to current acceptance rate (default = 0.95)
        """
        acceptance = self.move.performance_tracker.acceptance_rate

        if acceptance > 0.3 and acceptance <= 0.45:
            self.move.amp_bead *= factor
        elif acceptance >= 0.55 and acceptance < 0.7:
            self.move.amp_bead /= factor


def all_moves(
    log_dir: str,
    controller: Optional[Controller] = NoControl
) -> List[Controller]:
    """Generate a list of controllers for all adaptable MC moves.

    Parameters
    ----------
    log_dir : str
        Path to the directory in which to save log files
    controller : Optional[Controller]
        Bead and move amplitude controller (default = NoControl)
    
    Returns
    -------
    List of controllers for all adaptable MC moves.
    """
    return [
        controller(
            mv.MCAdapter(str(log_dir) + '/' + move.__name__, move)
        ) for move in [
            mv.crank_shaft, mv.end_pivot, mv.slide, mv.tangent_rotation
        ]
    ]
