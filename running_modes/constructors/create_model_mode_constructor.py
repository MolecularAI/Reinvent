from dacite import from_dict

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope, CreateModelConfiguration
from running_modes.create_model.create_model import CreateModelRunner
from running_modes.utils.general import set_default_device_cuda


class CreateModelModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        config = from_dict(data_class=CreateModelConfiguration, data=self._configuration.parameters)
        set_default_device_cuda()
        runner = CreateModelRunner(self._configuration, config)
        return runner