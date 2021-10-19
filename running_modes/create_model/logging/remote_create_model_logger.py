import requests

from running_modes.create_model.logging.base_create_model_logger import BaseCreateModelLogger
from running_modes.configurations.logging import get_remote_logging_auth_token


class RemoteCreateModelLogger(BaseCreateModelLogger):

    def log_message(self, message: str):
        data = {"Message": message}
        # TODO: utils endpoint should be scpecified in a separate configuration file
        self._notify_server(data, f"{self._log_config.recipient}/jobLog/log-id/{self._log_config.job_id}")

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._common_logger.warning(f"posting to {to_address}")
            headers = {
                'Accept': 'application/json', 'Content-Type': 'application/json',
                'Authorization': get_remote_logging_auth_token()
            }
            response = requests.post(to_address, json=data, headers=headers)

            if response.status_code == requests.codes.ok:
                self._common_logger.warning(f"SUCCESS: {response.status_code}")
            else:
                self._common_logger.warning(f"PROBLEM: {response.status_code}")
        except Exception as e:
            self._common_logger.error("Exception occurred", exc_info=True)
            self._common_logger.error(f"Attempted posting the following data:")
            self._common_logger.error(data, exc_info=False)
