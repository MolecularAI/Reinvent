import abc

from running_modes.reinforcement_learning.logging.local_reinforcement_logger import LocalReinforcementLogger


class BaseAction(abc.ABC):
    def __init__(self, logger=None):
        """
        (Abstract) Initializes an action.
        :param logger: An optional logger instance.
        """
        self.logger: LocalReinforcementLogger = logger
