from dacite import from_dict
from reinvent_models.reinvent_core.models.model import Model

from running_modes.automated_curriculum_learning.automated_curriculum_runner import AutomatedCurriculumRunner
from running_modes.automated_curriculum_learning.logging import AutoCLLogger
from running_modes.configurations import GeneralConfigurationEnvelope, AutomatedCLConfiguration
from running_modes.configurations.automated_curriculum_learning.base_configuration import BaseConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.curriculum_learning.curriculum_runner import CurriculumRunner
from running_modes.enums.curriculum_type_enum import CurriculumTypeEnum
from running_modes.utils.general import set_default_device_cuda


class CurriculumLearningModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        cl_enum = CurriculumTypeEnum
        base_config = from_dict(data_class=BaseConfiguration, data=self._configuration.parameters)
        set_default_device_cuda()

        if base_config.curriculum_type == cl_enum.MANUAL:
            runner = CurriculumRunner(self._configuration)
        else:
            self._configuration = configuration
            config = from_dict(data_class=AutomatedCLConfiguration, data=self._configuration.parameters)
            _logger = AutoCLLogger(self._configuration)
            _prior = Model.load_from_file(config.prior)
            _agent = Model.load_from_file(config.agent)
            runner = AutomatedCurriculumRunner(config, _logger, _prior, _agent)

        return runner