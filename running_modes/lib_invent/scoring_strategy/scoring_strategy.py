from running_modes.lib_invent.configurations.scoring_strategy_configuration import ScoringStrategyConfiguration
from running_modes.lib_invent.scoring_strategy.base_scoring_strategy import BaseScoringStrategy
from running_modes.lib_invent.scoring_strategy.scoring_strategy_enum import ScoringStrategyEnum
from running_modes.lib_invent.scoring_strategy.standard_strategy import StandardScoringStrategy


class ScoringStrategy:

    def __new__(cls, strategy_configuration: ScoringStrategyConfiguration, logger) -> BaseScoringStrategy:
        scoring_strategy_enum = ScoringStrategyEnum()
        if scoring_strategy_enum.STANDARD == strategy_configuration.name:
            return StandardScoringStrategy(strategy_configuration, logger)
