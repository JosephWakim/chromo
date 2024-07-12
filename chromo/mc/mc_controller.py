"""Control MC simulation parameters.
"""

# Built-in Modules
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Callable

# External Modules
# import numpy as np

# Custom modules
from chromo.polymers import PolymerBase
import chromo.mc.moves as mv


class Controller(ABC):
    """Class representation of controller for MC simulation parameters.
    """

    def __init__(
        self,
        mc_adapter: mv.MCAdapter,
        bead_amp_bounds: Tuple[int, int],
        move_amp_bounds: Tuple[float, float]
    ):
        """Initialize the MC controller.

        Parameters
        ----------
        mc_adapter : mv.MCAdapter
            Monte Carlo move adapter affected by controller
        bead_amp_bounds : Tuple[int, int]
            Bounds of bead selection amplitude permitted by controller in the
            form (minimum bead amplitude, maximum bead amplitude)
        move_amp_bounds : Tuple[float, float]
            Bounds for move amplitudes permitted by controller in the form
            (minimum move amplitude, maximum move amplitude)
        """
        if bead_amp_bounds[0] > bead_amp_bounds[1]:
            raise ValueError(
                "Lower bead amplitude bound must be less than \
                upper bead amplitude bound"
            )
        if move_amp_bounds[0] > move_amp_bounds[1]:
            raise ValueError(
                "Lower move amplitude bound must be less than \
                upper move amplitude bound"
            )
        self.move = mc_adapter
        self.bead_amp_bounds = bead_amp_bounds
        self.move_amp_bounds = move_amp_bounds

    @abstractmethod
    def update_amplitudes(self):
        """Update bead selection and move amplitudes.
        """
        pass

    @abstractmethod
    def update_move_amplitude(self):
        """Update the move amplitude.

        Use acceptance rate `self.move.acceptance_tracker.acceptance_rate`
        to update the move amplitude.
        """
        pass

    @abstractmethod
    def update_bead_amplitude(self):
        """Update the bead selection amplitude.

        Use acceptance rate `self.move.acceptance_tracker.acceptance_rate`
        to update the move amplitude.
        """
        pass


class NoControl(Controller):
    """Class representation of a trivial, non-acting MC controller.
    """
    def update_amplitudes(self):
        """Trivially maintain current move and bead amplitudes.
        """
        return

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

    A factor change is applied to the move amplitude based on the acceptance
    rate – if the acceptance rate is less than a setpoint (default 0.5), move
    amplitude is increased; otherwise, the move amplitude is decreased. Once
    the move amplitude reaches an upper or lower threshold, rather than
    changing the move amplitude, the bead amplitude is increased or decreased
    by a certain length, and the move amplitude is reset to its lowest value.
    """

    def __init__(
        self,
        mc_adapter: mv.MCAdapter,
        bead_amp_bounds: Tuple[int, int],
        move_amp_bounds: Tuple[float, float]
    ):
        """Initialize simple controller.
        """
        super(SimpleControl, self).__init__(
            mc_adapter=mc_adapter,
            bead_amp_bounds=bead_amp_bounds,
            move_amp_bounds=move_amp_bounds
        )
        self.name = "SimpleController"

    def to_file(self, path):
        pass

    def update_amplitudes(
        self,
        setpoint_acceptance: Optional[float] = 0.5,
        move_adjust_factor: Optional[float] = 0.95,
        num_delta_beads: Optional[int] = 1
    ):
        """ Update move and bead amplitudes using a simple controller.

        Parameters
        ----------
        setpoint_acceptance : Optional[float]
            Optional acceptance rate setpoint (default = 0.5)
        move_adjust_factor : Optional[float]
            Factor by which to multiply or divide move amplitude in response
            to current acceptance rate, between 0 and 1 (default = 0.95)
        num_delta_beads : Optional[int]
            Number of beads by which to adjust bead amplitude (default = 1)
        """
        self.update_move_amplitude(
            setpoint_acceptance, move_adjust_factor, num_delta_beads
        )

    def update_move_amplitude(
        self,
        setpoint_acceptance: Optional[float] = 0.5,
        move_adjust_factor: Optional[float] = 0.95,
        num_delta_beads: Optional[int] = 1
    ):
        """ Update move amplitude based on acceptance rate.

        Parameters
        ----------
        setpoint_acceptance : Optional[float]
            Optional acceptance rate setpoint (default = 0.5)
        move_adjust_factor : Optional[float]
            Factor by which to multiply or divide move amplitude in response
            to current acceptance rate, between 0 and 1 (default = 0.95)
        num_delta_beads : Optional[int]
            Number of beads by which to adjust bead amplitude (default = 1)
        """
        acceptance = self.move.acceptance_tracker.acceptance_rate

        if acceptance < setpoint_acceptance:
            prop_move_amp = self.move.amp_move * move_adjust_factor
            if prop_move_amp > self.move_amp_bounds[0]:
                self.move.amp_move = prop_move_amp
            else:
                self.move.amp_bead = max(
                    self.bead_amp_bounds[0],
                    self.update_bead_amplitude(
                        increase=False, num_delta_beads=num_delta_beads
                    )
                )
        elif acceptance > setpoint_acceptance:
            prop_move_amp = self.move.amp_move / move_adjust_factor
            if prop_move_amp < self.move_amp_bounds[1]:
                self.move.amp_move = prop_move_amp
            else:
                self.move.amp_bead = min(
                    self.bead_amp_bounds[1],
                    self.update_bead_amplitude(
                        increase=True, num_delta_beads=num_delta_beads
                    )
                )

    def update_bead_amplitude(
        self, increase: bool, num_delta_beads: Optional[int] = 1
    ):
        """Update the bead amplitude based on the acceptance rate.

        Parameters
        ----------
        increase : bool
            True or false indicator for whether to increase bead amplitude
        num_delta_bead: Optional[int]
            Number of beads by which to increase bead amplitude (default = 1)

        Returns
        -------
        prop_bead_amp : int
            Proposed bead amplitude after adjustment
        """
        if increase:
            self.move.amp_move = self.move_amp_bounds[0]
            return self.move.amp_bead + num_delta_beads
        else:
            self.move.amp_move = self.move_amp_bounds[1]
            return self.move.amp_bead - num_delta_beads


def all_moves(
    log_dir: str,
    bead_amp_bounds: Dict[str, Tuple[float, float]],
    move_amp_bounds: Dict[str, Tuple[float, float]],
    controller: Optional[Controller] = NoControl
) -> List[Controller]:
    """Generate a list of controllers for all adaptable MC moves (single poly).

    Parameters
    ----------
    log_dir : str
        Path to the directory in which to save log files
    bead_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of bead selection amplitude bounds where keys specify the
        name of the MC move and the values specify the selection amplitude
        bounds for the move in the form (lower bound, upper bound)
    move_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of maximum move amplitude bounds where keys specify the
        name of the MC move and the values specify the move amplitude bounds
        for the move in the form (lower bound, upper bound)
    controller : Optional[Controller]
        Bead and move amplitude controller (default = NoControl)

    Returns
    -------
    List of controllers for all adaptable MC moves.
    """
    controllers = [
        controller(
            mv.MCAdapter(
                str(log_dir) + '/acceptance_trackers',
                move.__name__ + "_snap_",
                move,
                moves_in_average=20,
                init_amp_bead=bead_amp_bounds[move.__name__][0],
                init_amp_move=move_amp_bounds[move.__name__][0]
            ),
            bead_amp_bounds=bead_amp_bounds[move.__name__],
            move_amp_bounds=move_amp_bounds[move.__name__]
        ) for move in [
            mv.crank_shaft, mv.end_pivot, mv.slide, mv.tangent_rotation,
            mv.change_binding_state
        ]
    ]
    controllers[0].move.num_per_cycle = 30
    controllers[1].move.num_per_cycle = 1
    controllers[2].move.num_per_cycle = 60
    controllers[3].move.num_per_cycle = 60
    controllers[4].move.num_per_cycle = 10

    return controllers


def all_moves_except_binding_state(
    log_dir: str,
    bead_amp_bounds: Dict[str, Tuple[float, float]],
    move_amp_bounds: Dict[str, Tuple[float, float]],
    controller: Optional[Controller] = NoControl
) -> List[Controller]:
    """Generate list of controllers for physical MC moves (single poly).

    Parameters
    ----------
    log_dir : str
        Path to the directory in which to save log files
    bead_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of bead selection amplitude bounds where keys specify the
        name of the MC move and the values specify the selection amplitude
        bounds for the move in the form (lower bound, upper bound)
    move_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of maximum move amplitude bounds where keys specify the
        name of the MC move and the values specify the move amplitude bounds
        for the move in the form (lower bound, upper bound)
    controller : Optional[Controller]
        Bead and move amplitude controller (default = NoControl)

    Returns
    -------
    List of controllers for all adaptable MC moves.
    """
    controllers = [
        controller(
            mv.MCAdapter(
                str(log_dir) + '/acceptance_trackers',
                move.__name__ + "_snap_",
                move,
                moves_in_average=20,
                init_amp_bead=bead_amp_bounds[move.__name__][0],
                init_amp_move=move_amp_bounds[move.__name__][0]
            ),
            bead_amp_bounds=bead_amp_bounds[move.__name__],
            move_amp_bounds=move_amp_bounds[move.__name__]
        ) for move in [
            mv.crank_shaft,
            mv.end_pivot,
            mv.slide,
            mv.tangent_rotation
        ]
    ]
    controllers[0].move.num_per_cycle = 5
    controllers[1].move.num_per_cycle = 5
    controllers[2].move.num_per_cycle = 5
    controllers[3].move.num_per_cycle = 5

    return controllers


def specific_move(
    move_fxn,
    log_dir: str,
    bead_amp_bounds: Dict[str, Tuple[float, float]],
    move_amp_bounds: Dict[str, Tuple[float, float]],
    controller: Optional[Controller] = NoControl,
) -> List[Controller]:
    """Generate a list of controllers for a specific MC move.

    Parameters
    ----------
    move_fxn : Callable
        Monte Carlo move function (from `move_funcs.pyx`)
    log_dir : str
        Path to the directory in which to save log files
    bead_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of bead selection amplitude bounds where keys specify the
        name of the MC move and the values specify the selection amplitude
        bounds for the move in the form (lower bound, upper bound)
    move_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of maximum move amplitude bounds where keys specify the
        name of the MC move and the values specify the move amplitude bounds
        for the move in the form (lower bound, upper bound)
    controller : Optional[Controller]
        Bead and move amplitude controller (default = NoControl)

    Returns
    -------
    List of controllers for all adaptable MC moves.
    """
    controllers = [
        controller(
            mv.MCAdapter(
                str(log_dir) + '/acceptance_trackers',
                move.__name__ + "_snap_",
                move,
                moves_in_average=20,
                init_amp_bead=bead_amp_bounds[move.__name__][0],
                init_amp_move=move_amp_bounds[move.__name__][0]
            ),
            bead_amp_bounds=bead_amp_bounds[move.__name__],
            move_amp_bounds=move_amp_bounds[move.__name__]
        ) for move in [
            move_fxn
        ]
    ]
    controllers[0].move.num_per_cycle = 1

    return controllers


def specific_moves(
        move_fxns: List[Callable],
        log_dir: str,
        bead_amp_bounds: Dict[str, Tuple[float, float]],
        move_amp_bounds: Dict[str, Tuple[float, float]],
        controller: Optional[Controller] = NoControl,
) -> List[Controller]:
    """Generate a list of controllers for all adaptable physical MC moves.

    Parameters
    ----------
    move_fxns : List[Callable]
        List of Monte Carlo move functions (from `move_funcs.pyx`)
    log_dir : str
        Path to the directory in which to save log files
    bead_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of bead selection amplitude bounds where keys specify the
        name of the MC move and the values specify the selection amplitude
        bounds for the move in the form (lower bound, upper bound)
    move_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of maximum move amplitude bounds where keys specify the
        name of the MC move and the values specify the move amplitude bounds
        for the move in the form (lower bound, upper bound)
    controller : Optional[Controller]
        Bead and move amplitude controller (default = NoControl)

    Returns
    -------
    List of controllers for all adaptable MC moves.
    """
    controllers = [
        controller(
            mv.MCAdapter(
                str(log_dir) + '/acceptance_trackers',
                move.__name__ + "_snap_",
                move,
                moves_in_average=20,
                init_amp_bead=bead_amp_bounds[move.__name__][0],
                init_amp_move=move_amp_bounds[move.__name__][0]
            ),
            bead_amp_bounds=bead_amp_bounds[move.__name__],
            move_amp_bounds=move_amp_bounds[move.__name__]
        ) for move in move_fxns
    ]
    for i in range(len(move_fxns)):
        controllers[i].move.num_per_cycle = 1

    return controllers


def only_binding_move(
    log_dir: str,
    bead_amp_bounds: Dict[str, Tuple[float, float]],
    move_amp_bounds: Dict[str, Tuple[float, float]],
    controller: Optional[Controller] = NoControl
) -> List[Controller]:
    """Generate list containing controller for the binding move (single poly).

    Parameters
    ----------
    log_dir : str
        Path to the directory in which to save log files
    bead_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of bead selection amplitude bounds where keys specify the
        name of the MC move and the values specify the selection amplitude
        bounds for the move in the form (lower bound, upper bound)
    move_amp_bounds : Dict[str, Tuple[float, float]]
        Dictionary of maximum move amplitude bounds where keys specify the
        name of the MC move and the values specify the move amplitude bounds
        for the move in the form (lower bound, upper bound)
    controller : Optional[Controller]
        Bead and move amplitude controller (default = NoControl)

    Returns
    -------
    List of controllers for all adaptable MC moves.
    """
    controllers = [
        controller(
            mv.MCAdapter(
                str(log_dir) + '/acceptance_trackers',
                move.__name__ + "_snap_",
                move,
                moves_in_average=20,
                init_amp_bead=bead_amp_bounds[move.__name__][0],
                init_amp_move=move_amp_bounds[move.__name__][0]
            ),
            bead_amp_bounds=bead_amp_bounds[move.__name__],
            move_amp_bounds=move_amp_bounds[move.__name__]
        ) for move in [
            mv.change_binding_state
        ]
    ]
    controllers[0].move.num_per_cycle = 5
    return controllers
