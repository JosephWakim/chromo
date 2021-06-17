"""Utilities for tracking MC move acceptance rates.
"""

# Built-in Modules
from abc import ABC, abstractmethod
import csv
from typing import List

# External Modules
import numpy as np


class Tracker(ABC):
    """Abstract class representation of a MC performance tracker.
    """

    @abstractmethod
    def create_log_file(self):
        """Create an output file for the tracker.
        """
        pass

    @abstractmethod
    def log_move(self):
        """Log performance after an MC move.
        """
        pass

    @abstractmethod
    def save_move_log(self):
        """Save the log to the output file, then reset log.
        """
        pass


class AcceptanceTracker(Tracker):
    """Class representation of MC acceptance rate tracker.

    Tracks acceptance rate of attempted MC moves of a particular type.
    Downweights historical acceptance rates with each step to more heavily
    consider recent acceptance rates. Logs the acceptance rate, move amplitude,
    and bead selection amplitude with each step.

    This class will provide metrics to the MC adapter, which will dynamically
    adjust move and bead amplitudes.
    """

    def __init__(self, log_path: str, moves_in_average: float):
        """Initialize the performance tracker object.

        Move acceptance rate will be tracked as an "N-day" exponentially
        weighted moving average (EWMA), where N gives the number of moves to
        track in the average, specified by `moves_in_average`. This results in
        a decay factor `alpha` = 2 / (`moves_in_average` + 1).

        Parameters
        ----------
        log_path : str
            Path to log file tracking bead/move amplitudes and acceptance rate
        moves_in_average : float
            Number of historical moves to track in incremental measure of move
            acceptance rate
        """
        self.log_path = log_path
        self.alpha = 2 / (moves_in_average + 1)
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
        column_labels = [
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
        initialize_table(self.log_path, column_labels)

    def log_move(
        self,
        amp_move_limit: float,
        amp_bead_limit: int,
        amp_move: float,
        amp_bead: int,
        dE: float
    ):
        """Add a proposed move and current acceptance rate to the log.

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

    def update_acceptance_rate(self, accept: bool):
        """Incrementally update acceptance rate based on move acceptance.

        Apply an exponentially weighted moving average to maintain a running
        measure of move acceptance weight.

        Parameters
        ----------
        accept : bool
            Binary indicator of move acceptance (True) or rejection (False)
        """
        self.move_accepted.append(int(accept))
        self.acceptance_rate = (
            self.alpha * int(accept)
        ) + (1 - self.alpha) * self.acceptance_rate


class ConfigurationTracker(Tracker):
    """Track progressive reconfiguraion of the polymer.
    """

    def __init__(self, log_path: str, initial_config: np.ndarray):
        """Initialize the `ConfigurationTracker` object.

        Parameters
        ----------
        log_path : str
            Path to the output file for confiruation tracker
        initial_config : np.ndarray (N, 3)
            Initial polymer configuration; rows represent individual beads and
            columns represent (x, y, z) coordinates
        """
        self.log_path = log_path
        self.previous_config = initial_config
        self.N = initial_config.shape[0]
        self.RMSD = 0
        self.RMSD_log = []

    def create_log_file(self):
        column_labels = [
            "snapshot",
            "iteration",
            "step_RMSD"
        ]
        initialize_table(self.log_path, column_labels)

    def log_move(self, config: np.ndarray):
        """Log the change in polymer configuration.

        Parameters
        ----------
        config : np.ndarray (N, 3)
            Current polymer configuration; rows represent individual beads and
            columns represent (x, y, z) coordinates
        """
        if config.shape[0] != self.N:
            raise ValueError(
                "The shape of the table representing current polymer \
                configuration is inconsistent with that of the previous \
                configuration."
            )
        RMSD = np.sqrt(
            1 / self.N * np.sum(
                np.linalg.norm(config - self.previous_config, ord=1, axis=1)
            )
        )
        self.RMSD = RMSD
        self.RMSD_log.append(RMSD)
        self.previous_config = config

    def save_move_log(self, snapshot: int):
        """Save to the `ConfigurationTracker` output file and reset log.

        Parameters
        ----------
        snapshot : int
            MC snapshot number
        """
        num_iterations = len(self.RMSD_log)
        with open(self.log_path, 'a') as output:
            w = csv.writer(output, delimiter=',')
            for i in range(num_iterations):
                row = [
                    snapshot,
                    i+1,
                    self.RMSD_log[i],
                ]
                w.writerow(row)

        self.RMSD_log = []


def initialize_table(path: str, columns: List[str]):
    """Initialize a log file with empty columns.

    Parameters
    ----------
    path : str
        Path to the log file being initiailzed
    columns : List[str]
        Column names for log file
    """
    output = open(path, 'w')
    w = csv.writer(output, delimiter=',')
    w.writerow(columns)
    output.close()
