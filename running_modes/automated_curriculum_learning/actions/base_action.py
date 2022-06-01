import abc
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger


class BaseAction(abc.ABC):
    def __init__(self, logger=None):
        """
        (Abstract) Initializes an action.
        :param logger: An optional logger instance.
        """
        self.logger: BaseLogger = logger
