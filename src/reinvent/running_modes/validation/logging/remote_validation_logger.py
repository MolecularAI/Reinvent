import requests

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.validation.logging.base_validation_logger import BaseValidationLogger


class RemoteValidationLogger(BaseValidationLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)

    def log_message(self, message: str):
        data = {"valid": self.model_is_valid, "message": message}
        self._notify_server(data, self._log_config.recipient)

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._common_logger.warning(f"posting to {to_address}")
            response = requests.post(to_address, data=data)

            if response.status_code == requests.codes.ok:
                self._common_logger.info(f"SUCCESS: {response.status_code}")
            else:
                self._common_logger.info(f"PROBLEM: {response.status_code}")
        except Exception as e:
            self._common_logger.exception("Exception occurred", exc_info=True)
            self._common_logger.exception(f"Attempted posting the following data:")
            self._common_logger.exception(data, exc_info=False)
