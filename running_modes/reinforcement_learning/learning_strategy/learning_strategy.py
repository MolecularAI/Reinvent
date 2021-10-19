from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import LearningStrategyConfiguration
from running_modes.reinforcement_learning.learning_strategy import BaseLearningStrategy
from running_modes.reinforcement_learning.learning_strategy import DAPStrategy
from running_modes.reinforcement_learning.learning_strategy import LearningStrategyEnum
from running_modes.reinforcement_learning.learning_strategy import MASCOFStrategy
from running_modes.reinforcement_learning.learning_strategy import MAULIStrategy
from running_modes.reinforcement_learning.learning_strategy import SDAPStrategy


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
