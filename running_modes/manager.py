import os

from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.constructors.running_mode import RunningMode
from running_modes.enums.running_mode_enum import RunningModeEnum


class Manager:

    def __init__(self, base_configuration, run_configuration):
        self.running_mode_enum = RunningModeEnum()
        self.base_configuration = base_configuration
        self.run_configuration = GeneralConfigurationEnvelope(**run_configuration)
        self._load_environmental_variables()

    def run(self):
        runner = RunningMode(self.run_configuration)
        runner.run()

    def _load_environmental_variables(self):
        environmental_variables = self.base_configuration["ENVIRONMENTAL_VARIABLES"]
        for key, value in environmental_variables.items():
            os.environ[key] = value
