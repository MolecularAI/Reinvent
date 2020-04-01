import json
import logging
import os
from abc import ABC, abstractmethod

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.scoring_log_configuration import ScoringLoggerConfiguration
from scoring.score_summary import FinalSummary


class BaseScoringLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self._configuration = configuration
        self._log_config = ScoringLoggerConfiguration(**self._configuration.logging)
        self._logger = self._setup_logger()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    @abstractmethod
    def log_results(self, score_summary: FinalSummary):
        raise NotImplementedError("log_results method is not implemented")

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("scoring_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger