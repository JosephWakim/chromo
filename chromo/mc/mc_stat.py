"""Utilities for tracking MC move acceptance rates.
"""

# Built-in Modules
import csv
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

    def __init__(self, log_path: str, decay_rate: float):
        """Initialize the performance tracker object.

        Parameters
        ----------
        log_path : str
            Path to log file tracking bead/move amplitudes and acceptance rate
        decay_rate : float
            Factor by which to downweight historical acceptance rate in mean
        """
        self.log_path = log_path
        self.decay_rate = decay_rate

        self.N = 0
        self.acceptance_rate = 0
        self.amp_bead_limit_log: List[float] = []
        self.amp_move_limit_log: List[float] = []
        self.amp_bead_realized_log: List[float] = []
        self.amp_move_realized_log: List[float] = []
        self.dE_log = []
        self.move_accepted = []
        self.acceptance_log: List[float] = []

        self.create_log_file()

    def create_log_file(self):
        """Create the log file tracking amplitudes and acceptance rates.

        The log file will have five columns: snapshot count, iteration count
        bead selection amplitude, move amplitude, and move acceptance rate.

        Snapshot count and will be recorded as constant values for each
        snapshot. The iteration count, bead selection amplitude, move
        amplitude, and move acceptance rate will be recorded for each iteration
        in the snapshot.

        This method simply creates the log file and adds the column labels. If
        additional properties are to be logged, add them as column labels in
        this method, and output them using the `save_move_log` method.
        """
        row_labels = [
            "snapshot",
            "iteration",
            "bead_amp_limit",
            "move_amp_limit",
            "bead_amp_realized",
            "move_amp_realized",
            "dE",
            "accepted",
            "acceptance_rate"
        ]
        output = open(self.log_path, 'w')
        w = csv.writer(output, delimiter=',')
        w.writerow(row_labels)
        output.close()

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
        self.move_accepted.append(int(accept))
        self.N += 1
        self.acceptance_rate = (
            self.acceptance_rate * (9) + int(accept)
        ) / (
            10
        )

    def log_move(
        self,
        amp_move_limit: float,
        amp_bead_limit: int,
        amp_move: float,
        amp_bead: int,
        dE: float
    ):
        """Add a proposed move to the log.

        Parameters
        ----------
        amp_move_limit : float
            Maximum move amplitude allowed at current MC step
        amp_bead_limit : int
            Maximum bead selection amplitude allowed at current MC step
        amp_move : float
            Amplitude of the proposed move
        amp_bead : int
            Selection amplitude of the proposed move
        dE : float
            Change in energy associated with move
        """
        self.amp_move_limit_log.append(amp_move_limit)
        self.amp_bead_limit_log.append(amp_bead_limit)
        self.amp_move_realized_log.append(amp_move)
        self.amp_bead_realized_log.append(amp_bead)
        self.dE_log.append(dE)

    def log_acceptance_rate(self):
        """Log average acceptance rate, while decaying historical values.
        """
        self.acceptance_log.append(self.acceptance_rate)

    def save_move_log(self, snapshot: int):
        """Save the move and bead amplitude log to a file and clear lists.

        Parameters
        ----------
        snapshot : int
            Current snapshot number, with which to label
        """
        num_iterations = len(self.amp_move_realized_log)
        if num_iterations != len(self.amp_bead_realized_log):
            raise ValueError(
                "The number of bead selection amplitudes logged does not match \
                the number of move amplitudes logged."
            )
        if num_iterations != len(self.acceptance_log):
            raise ValueError(
                "The number of move acceptance rates logged does not match the \
                number of move amplitudes logged."
            )
        with open(self.log_path, 'a') as output:
            w = csv.writer(output, delimiter=',')
            for i in range(num_iterations):
                row = [
                    snapshot,
                    i+1,
                    self.amp_bead_limit_log[i],
                    self.amp_move_limit_log[i],
                    self.amp_bead_realized_log[i],
                    self.amp_move_realized_log[i],
                    self.dE_log[i],
                    self.move_accepted[i],
                    self.acceptance_log[i]
                ]
                w.writerow(row)

        self.amp_move_limit_log = []
        self.amp_bead_limit_log = []
        self.amp_move_realized_log = []
        self.amp_bead_realized_log = []
        self.dE_log = []
        self.move_accepted = []
        self.acceptance_log = []
