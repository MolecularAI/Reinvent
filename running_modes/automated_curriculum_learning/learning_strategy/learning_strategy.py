from running_modes.automated_curriculum_learning.learning_strategy import DAPStrategy, MAULIStrategy, MASCOFStrategy, \
    SDAPStrategy, DAPSingleQueryStrategy
from running_modes.automated_curriculum_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_enum import LearningStrategyEnum


class LearningStrategy:

    def __new__(cls, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None) \
            -> BaseLearningStrategy:
        learning_strategy_enum = LearningStrategyEnum()
        if learning_strategy_enum.DAP_SINGLE_QUERY == configuration.name:
            return DAPSingleQueryStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.DAP == configuration.name:
            return DAPStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.MAULI == configuration.name:
            return MAULIStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.MASCOF == configuration.name:
            return MASCOFStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.SDAP == configuration.name:
            return SDAPStrategy(critic_model, optimizer, configuration, logger)
