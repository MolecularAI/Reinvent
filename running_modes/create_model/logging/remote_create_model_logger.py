import logging
import sys

import requests

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration


class RemoteCreateModelLogger:
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        #TODO: input config should be stored as a JSON in the results folder
        self.configuration = configuration
        self._log_config = CreateModelLoggerConfiguration(**self.configuration.logging)
        self._common_logger = self._setup_logger(name="create_model_logger")

    def log_message(self, message: str):
        data = {"Message": message}
        # TODO: utils endpoint should be scpecified in a separate configuration file
        self._notify_server(data, f"{self._log_config.recipient}/jobLog/log-id/{self._log_config.job_id}")

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._common_logger.warning(f"posting to {to_address}")
            response = requests.post(to_address, data=data)

            if response.status_code == requests.codes.ok:
                self._common_logger.warning(f"SUCCESS: {response.status_code}")
            else:
                self._common_logger.warning(f"PROBLEM: {response.status_code}")
        except Exception as e:
            self._common_logger.error("Exception occurred", exc_info=True)
            self._common_logger.error(f"Attempted posting the following data:")
            self._common_logger.error(data, exc_info=False)

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
