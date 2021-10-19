from abc import ABC, abstractmethod

from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger


class BaseAction(ABC):
    def __init__(self, logger):
        """
        (Abstract) Initializes an action.
        :param logger: An optional logger instance.
        """
        self._logger: BaseTransferLearningLogger = logger

    @abstractmethod
    def run(self):
        raise NotImplementedError('Method not implemented')
