import json
import os

import requests

import utils.logging.log as utils_log
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.scoring.logging.base_scoring_logger import BaseScoringLogger
from scoring.score_summary import FinalSummary


class RemoteScoringLogger(BaseScoringLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._is_dev = utils_log._is_development_environment()

    def log_message(self, message: str):
        self._logger.info(message)

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def log_results(self, score_summary: FinalSummary):
        # TODO this should notify the server
        pass

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._logger.warning(f"posting to {to_address}")
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            response = requests.post(to_address, json=data, headers=headers)

            if self._is_dev:
                """logs out the response content only when running a test instance"""
                if response.status_code == requests.codes.ok:
                    self._logger.info(f"SUCCESS: {response.status_code}")
                    self._logger.info(response.content)
                else:
                    self._logger.info(f"PROBLEM: {response.status_code}")
                    self._logger.exception(data, exc_info=False)
        except Exception as t_ex:
            self._logger.exception("Exception occurred", exc_info=True)
            self._logger.exception(f"Attempted posting the following data:")
            self._logger.exception(data, exc_info=False)