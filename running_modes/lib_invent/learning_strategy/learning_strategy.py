from running_modes.lib_invent.configurations.learning_strategy_configuration import LearningStrategyConfiguration
from running_modes.lib_invent.learning_strategy.base_learning_strategy import BaseLearningStrategy
from running_modes.lib_invent.learning_strategy.dap_strategy import DAPStrategy
from running_modes.lib_invent.learning_strategy.learning_strategy_enum import LearningStrategyEnum
from running_modes.lib_invent.learning_strategy.mascof_strategy import MASCOFStrategy
from running_modes.lib_invent.learning_strategy.mauli_strategy import MAULIStrategy
from running_modes.lib_invent.learning_strategy.sdap_strategy import SDAPStrategy


class LearningStrategy:

    def __new__(cls, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None) \
            -> BaseLearningStrategy:
        learning_strategy_enum = LearningStrategyEnum()
        if learning_strategy_enum.DAP == configuration.name:
            return DAPStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.MAULI == configuration.name:
            return MAULIStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.MASCOF == configuration.name:
            return MASCOFStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.SDAP == configuration.name:
            return SDAPStrategy(critic_model, optimizer, configuration, logger)
