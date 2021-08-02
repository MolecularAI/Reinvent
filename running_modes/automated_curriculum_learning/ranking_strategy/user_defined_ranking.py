from typing import Dict

from reinvent_scoring import ScoringFuncionParameters

from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.ranking_strategy.base_ranking_strategy import BaseRankingStrategy
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations import RankingStrategyConfiguration
from running_modes.dto.scoring_table_entry_dto import ScoringTableEntryDTO

from running_modes.reinforcement_learning.inception import Inception


class UserDefinedOrder(BaseRankingStrategy):
    def __init__(self, prior: Model, agent: Model, scoring_function_params: ScoringFuncionParameters,
                 diversity_filter: BaseDiversityFilter, inception: Inception,
                 configuration: RankingStrategyConfiguration,
                 logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        super().__init__(prior=agent, agent=prior, scoring_function_params=scoring_function_params,
                         diversity_filter=diversity_filter, inception=inception, configuration=configuration,
                         logger=logger, scoring_table=scoring_table)

    def train_agent(self, agent: Model, scoring_function_component: Dict):
        specific_params = scoring_function_component.get(self._special_parameters_enum.SPECIFIC_PARAMETERS, {})
        component_order = specific_params.get(self._special_parameters_enum.ORDER, 0)
        score = -component_order

        self._scoring_table.add_score_for_agent(
            ScoringTableEntryDTO(agent=agent, score=score, scoring_function_components=scoring_function_component))
        self._logger.log_message(
            f"Ranking Position: {component_order}"f". Component: {scoring_function_component.get('name')}")