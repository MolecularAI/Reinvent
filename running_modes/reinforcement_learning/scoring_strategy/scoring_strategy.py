from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter

from running_modes.reinforcement_learning.configurations.scoring_strategy_configuration import \
    ScoringStrategyConfiguration
from running_modes.reinforcement_learning.scoring_strategy.base_scoring_strategy import BaseScoringStrategy
from running_modes.reinforcement_learning.scoring_strategy.lib_invent_scoring_strategy import LibInventScoringStrategy
from running_modes.reinforcement_learning.scoring_strategy.link_invent_scoring_strategy import LinkInventScoringStrategy
from running_modes.reinforcement_learning.scoring_strategy.scoring_strategy_enum import ScoringStrategyEnum


class ScoringStrategy:

    def __new__(cls, strategy_configuration: ScoringStrategyConfiguration, diversity_filter: BaseDiversityFilter,
                logger) -> BaseScoringStrategy:
        scoring_strategy_enum = ScoringStrategyEnum()
        if scoring_strategy_enum.LINK_INVENT == strategy_configuration.name:
            return LinkInventScoringStrategy(strategy_configuration, diversity_filter, logger)
        elif scoring_strategy_enum.LIB_INVENT == strategy_configuration.name:
            #TODO: check the type passed to LibInventScoringStrategy
            return LibInventScoringStrategy(strategy_configuration, diversity_filter, logger)
        else:
            raise KeyError(f"Incorrect strategy name: `{strategy_configuration.name}` provided")

