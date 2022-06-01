from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel

from running_modes.automated_curriculum_learning.automated_curriculum_runner import AutomatedCurriculumRunner
from running_modes.automated_curriculum_learning.logging import AutoCLLogger
from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.configurations.automated_curriculum_learning.automated_curriculum_learning_input_configuration import \
    AutomatedCurriculumLearningInputConfiguration
from running_modes.configurations.automated_curriculum_learning.base_configuration import BaseConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.curriculum_learning.curriculum_runner import CurriculumRunner
from running_modes.enums.curriculum_type_enum import CurriculumTypeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.utils.general import set_default_device_cuda


class CurriculumLearningModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        cl_enum = CurriculumTypeEnum

        base_config = BaseConfiguration.parse_obj(self._configuration.parameters)

        if base_config.curriculum_type == cl_enum.MANUAL:
            set_default_device_cuda()
            runner = CurriculumRunner(self._configuration)
        elif base_config.curriculum_type == cl_enum.AUTOMATED:
            runner = self._create_automated_curriculum(self._configuration)
        else:
            raise KeyError(f"Incorrect curriculum type: `{base_config.curriculum_type}` provided")

        return runner

    @staticmethod
    def _create_automated_curriculum(configuration):
        model_type = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()

        if model_type.DEFAULT == configuration.model_type:
            set_default_device_cuda()
            config = AutomatedCurriculumLearningInputConfiguration.parse_obj(configuration.parameters)
        elif model_type.LINK_INVENT == configuration.model_type:
            set_default_device_cuda()
            config = AutomatedCurriculumLearningInputConfiguration.parse_obj(configuration.parameters)
        else:
            raise KeyError(f"Incorrect model type: `{configuration.model_type}` provided")

        _logger = AutoCLLogger(configuration)
        prior_config = ModelConfiguration(configuration.model_type, model_regime.INFERENCE, config.prior)
        agent_config = ModelConfiguration(configuration.model_type, model_regime.TRAINING, config.agent)
        _prior = GenerativeModel(prior_config)
        _agent = GenerativeModel(agent_config)

        runner = AutomatedCurriculumRunner(config, _logger, _prior, _agent)
        return runner
