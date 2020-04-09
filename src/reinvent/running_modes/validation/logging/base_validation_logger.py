import logging
import sys
from abc import ABC, abstractmethod

from running_modes.configurations import BaseLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope


class BaseValidationLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self.model_is_valid = False
        self._configuration = configuration
        self._log_config = BaseLoggerConfiguration(**self._configuration.logging)
        self._common_logger = self._setup_logger()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    def _setup_logger(self):
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("validation_logger")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
        return logger
