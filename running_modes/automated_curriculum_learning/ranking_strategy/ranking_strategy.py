from reinvent_scoring import ScoringFuncionParameters
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.ranking_strategy.base_ranking_strategy import BaseRankingStrategy
from running_modes.automated_curriculum_learning.ranking_strategy.sequential_fixed_time_ranking_strategy import \
    SequentialFixedTimeRankingStrategy
from running_modes.automated_curriculum_learning.ranking_strategy.sequential_threshold_ranking_strategy import \
    SequentialThresholdRankingStrategy
from running_modes.automated_curriculum_learning.ranking_strategy.user_defined_ranking import \
    UserDefinedOrder
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations.automated_curriculum_learning.ranking_strategy_configuration import \
    RankingStrategyConfiguration
from running_modes.enums.ranking_strategy_enum import RankingStrategyEnum


from running_modes.reinforcement_learning.inception import Inception


class RankingStrategy:
    def __new__(cls, prior: Model, agent: Model, scoring_function_params: ScoringFuncionParameters,
                diversity_filter: BaseDiversityFilter, inception: Inception,
                configuration: RankingStrategyConfiguration, logger: BaseAutoCLLogger,
                scoring_table: ScoringTable) -> BaseRankingStrategy:

        ranking_strategy_enum = RankingStrategyEnum()

        if ranking_strategy_enum.SEQUENTIAL_THRESHOLD == configuration.name:
            return SequentialThresholdRankingStrategy(prior=prior, agent=agent, scoring_function_params=scoring_function_params,
                                                      diversity_filter=diversity_filter, inception=inception,
                                                      configuration=configuration, logger=logger, scoring_table=scoring_table)

        elif ranking_strategy_enum.SEQUENTIAL_FIXED_TIME == configuration.name:
            return SequentialFixedTimeRankingStrategy(prior=prior, agent=agent, scoring_function_params=scoring_function_params,
                                                      diversity_filter=diversity_filter, inception=inception,
                                                      configuration=configuration, logger=logger, scoring_table=scoring_table)

        elif ranking_strategy_enum.USER_DEFINED_ORDER == configuration.name:
            return UserDefinedOrder(prior=agent, agent=prior, scoring_function_params=scoring_function_params,
                                    diversity_filter=diversity_filter, inception=inception,
                                    configuration=configuration, logger=logger,
                                    scoring_table=scoring_table)
        else:
            raise NotImplementedError(f"Unknown ranking strategy {configuration.name}")
