import json
import logging
import os
import sys

from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ...configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration


class CreateModelLogger:
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self.configuration = configuration
        self._log_config = CreateModelLoggerConfiguration(**self.configuration.logging)
        self._common_logger = self._setup_logger(name="create_model_logger")

    def log_message(self, message: str):
        self._common_logger.info(message)

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self.configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def _setup_logger(self, name, level=logging.INFO):
        logging.getLogger(name).addHandler(logging.NullHandler())
        handler = logging.StreamHandler(stream=sys.stderr)

        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(level)
            logger.addHandler(handler)
        return logger
