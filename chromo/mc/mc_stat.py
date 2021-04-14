"""Utilities for tracking MC move acceptance rates.
"""

# Built-in Modules
from typing import List


class PerformanceTracker:
    """Class representation of MC performance tracker.

    Tracks acceptance rate of attempted MC moves of a particular type.
    Downweights historical acceptance rates with each step to more heavily
    consider recent acceptance rates. Logs the acceptance rate, move amplitude,
    and bead selection amplitude with each step.

    This class will provide metrics to the MC adapter, which will dynamically
    adjust move and bead amplitudes.
    """

    def __init__(self, decay_rate: float):
        """Initialize the performance tracker object.

        Parameters
        ----------
        decay_rate : float
            Factor by which to downweight historical acceptance rate in mean
        """
        self.decay_rate = decay_rate

        self.N = 0
        self.acceptance_rate = 0
        self.amp_bead_log: List[float] = []
        self.amp_move_log: List[float] = []
        self.acceptance_log: List[float] = []

    def update_acceptance_rate(self, accept: bool):
        """Increment acceptance rate based on an accepted or rejected step

        Calculate a running, incremental average acceptance rate, which
        downweights historical values by a factor `self.decay_rate`.

        TODO: Verify weighted incremental average algorithm.

        Parameters
        ----------
        accept : bool
            Binary indicator of move acceptance (True) or rejection (False)
        """
        self.N += 1
        self.acceptance_rate = (
            self.acceptance_rate * (self.N - 1) * self.decay_rate + int(accept)
        ) / (
            (self.N - 1) * self.decay_rate + 1
        )

    def log_move(self, amp_move: float, amp_bead: float):
        """Add a proposed move to the log.

        Parameters
        ----------
        amp_move : float
            Amplitude of the proposed move
        amp_bead : float
            Selection amplitude of the proposed move
        accept : bool
            Indicator of whether or not the move was accepted
        """
        self.amp_move_log.append(amp_move)
        self.amp_bead_log.append(amp_bead)

    def log_acceptance_rate(self):
        """Log average acceptance rate, while decaying historical values.
        """
        self.acceptance_log.append(self.acceptance_rate)

    def save_move_log(
        self, path_move_log: str, path_bead_log: str, path_accept_log: str
    ):
        """Save the move and bead amplitude log to a file and clear lists.

        Parameters
        ----------
        path_move_log, path_bead_log, path_accept_log : str
            Path names at which to save move/bead amplitude logs and
            acceptance log.
        """
        with open(path_move_log, "w") as output:
            for val in self.amp_move_log:
                output.write(str(val) + '\n')
        with open(path_bead_log, "w") as output:
            for val in self.amp_bead_log:
                output.write(str(val) + '\n')
        with open(path_accept_log, "w") as output:
            for val in self.acceptance_log:
                output.write(str(val) + '\n')

        self.amp_move_log = []
        self.amp_bead_log = []
        self.acceptance_log = []
