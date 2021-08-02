import json
import os

from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.constructors.running_mode import RunningMode

class Manager:

    def __init__(self, configuration):
        self.running_mode_enum = RunningModeEnum()
        self.configuration = GeneralConfigurationEnvelope(**configuration)
        self._load_environmental_variables()

    def run(self):
        runner = RunningMode(self.configuration)
        runner.run()

    def _load_environmental_variables(self):
        try:
            project_root = os.path.dirname(__file__)
            with open(os.path.join(project_root, '../configs/config.json'), 'r') as f:
                config = json.load(f)
            environmental_variables = config["ENVIRONMENTAL_VARIABLES"]
            for key, value in environmental_variables.items():
                os.environ[key] = value

        except KeyError as ex:
            raise ex
