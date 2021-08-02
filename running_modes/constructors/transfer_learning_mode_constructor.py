from reinvent_models.reinvent_core.models.model import Model

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope, TransferLearningConfiguration
from running_modes.transfer_learning.transfer_learning_runner import TransferLearningRunner
from dacite import from_dict
from running_modes.utils import set_default_device_cuda


class TransferLearningModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        config = from_dict(data_class=TransferLearningConfiguration, data=self._configuration.parameters)
        set_default_device_cuda()
        model = Model.load_from_file(config.input_model_path)
        runner = TransferLearningRunner(model, config, self._configuration)
        return runner