import json
import os
import logging
from abc import ABC, abstractmethod
from dacite import from_dict

from running_modes.configurations import GeneralConfigurationEnvelope, CreateModelConfiguration
from running_modes.configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration


class BaseCreateModelLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self.configuration = configuration
        self._log_config = CreateModelLoggerConfiguration(**self.configuration.logging)
        self._common_logger = self._setup_logger(name="create_model_logger")

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    def log_out_input_configuration(self):
        config = from_dict(CreateModelConfiguration, self.configuration.parameters)
        config_save_path = os.path.join(os.path.dirname(config.output_model_path),
                                        f"{os.path.basename(config.output_model_path).split('.')[0]}_config.json")
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)

        jsonstr = json.dumps(self.configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(config_save_path, 'w') as f:
            f.write(jsonstr)


    def _setup_logger(self, name, level=logging.INFO):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(level)
        logger.propagate = False
        return logger

