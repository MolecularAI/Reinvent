from dacite import from_dict
from reinvent_scoring.scoring import ComponentParameters

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.validation.validation_runner import ValidationRunner


class ValidationModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        config = from_dict(data_class=ComponentParameters, data=self._configuration.parameters)
        runner = ValidationRunner(self._configuration, config)
        return runner