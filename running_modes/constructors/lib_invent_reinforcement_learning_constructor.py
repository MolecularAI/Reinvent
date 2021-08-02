from dacite import from_dict
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.lib_invent.models.model import DecoratorModel

from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.lib_invent.configurations.log_configuration import LogConfiguration
from running_modes.lib_invent.configurations.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration
from running_modes.lib_invent.lib_invent_reinforcement_learning import LibInventReinforcementLearning
from running_modes.lib_invent.logging.reinforcement_logger import ReinforcementLogger
from running_modes.utils.general import set_default_device_cuda


class LibInventReinforcementLearningModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        model_regime = GenerativeModelRegimeEnum()
        self._configuration = configuration
        config = from_dict(data_class=ReinforcementLearningConfiguration, data=self._configuration.parameters)
        logging_config = LogConfiguration(**configuration.logging)
        set_default_device_cuda()

        critic = DecoratorModel.load_from_file(config.critic, mode=model_regime.INFERENCE)
        actor = DecoratorModel.load_from_file(config.actor, mode=model_regime.TRAINING)
        logger = ReinforcementLogger(logging_config)
        runner = LibInventReinforcementLearning(critic=critic, actor=actor, configuration=config, logger=logger)
        return runner