from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.merging_strategy.base_merging_strategy import BaseMergingStrategy
from running_modes.automated_curriculum_learning.merging_strategy.linear_merging_strategy_with_removal import \
    LinearMergingStrategyWithRemoval
from running_modes.automated_curriculum_learning.merging_strategy.linear_merging_strategy_with_threshold import \
    LinearMergingStrategyWithThreshold
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations.automated_curriculum_learning.merging_strategy_configuration import \
    MergingStrategyConfiguration
from running_modes.enums.merging_strategy_enum import MergingStrategyEnum
from running_modes.reinforcement_learning.inception import Inception


class MergingStrategy:
    def __new__(cls, prior: Model, scoring_function_name: str, diversity_filter: BaseDiversityFilter,
                inception: Inception, configuration: MergingStrategyConfiguration, logger: BaseAutoCLLogger,
                scoring_table: ScoringTable) -> BaseMergingStrategy:
        merging_strategy_enum = MergingStrategyEnum()

        if merging_strategy_enum.LINEAR_MERGE_WITH_THRESHOLD == configuration.name:
            return LinearMergingStrategyWithThreshold(prior=prior, scoring_function_name=scoring_function_name,
                                                      configuration=configuration, diversity_filter=diversity_filter,
                                                      logger=logger, scoring_table=scoring_table, inception=inception)

        elif merging_strategy_enum.LINEAR_MERGE_WITH_REMOVAL == configuration.name:
            return LinearMergingStrategyWithRemoval(prior=prior, scoring_function_name=scoring_function_name,
                                                    configuration=configuration, diversity_filter=diversity_filter,
                                                    logger=logger, scoring_table=scoring_table, inception=inception)
        else:
            raise NotImplementedError(f"Unknown merging strategy name {configuration.name}")
