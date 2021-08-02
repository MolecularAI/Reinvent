import time
from typing import Dict
import torch
from reinvent_scoring import ScoringFuncionParameters
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.ranking_strategy.base_ranking_strategy import BaseRankingStrategy
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.dto.scoring_table_entry_dto import ScoringTableEntryDTO
from running_modes.configurations import RankingStrategyConfiguration

from running_modes.reinforcement_learning.inception import Inception


class SequentialThresholdRankingStrategy(BaseRankingStrategy):
    def __init__(self, prior: Model, agent: Model, scoring_function_params: ScoringFuncionParameters,
                 diversity_filter: BaseDiversityFilter, inception: Inception, configuration: RankingStrategyConfiguration,
                 logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        super().__init__(prior=prior, agent=agent, scoring_function_params=scoring_function_params,
                         diversity_filter=diversity_filter, inception=inception, configuration=configuration,
                         logger=logger, scoring_table=scoring_table)

        self.threshold_ranking = self._parameters.special_parameters.get(
            self._special_parameters_enum.THRESHOLD_RANKING)
        self.max_num_iteration = self._parameters.special_parameters.get(
            self._special_parameters_enum.MAX_NUM_ITERATIONS)

    def check_conditions(self, score: float, step: int) -> bool:
        condition_1 = score < self.threshold_ranking
        condition_2 = step < self.max_num_iteration
        condition = condition_1 and condition_2

        return condition

    def train_agent(self, agent: Model, scoring_function_component: Dict):
        optimiser = torch.optim.Adam(agent.network.parameters(), lr=self._parameters.learning_rate)
        scoring_function = self._setup_scoring_function(name=self._scoring_function_name, parameter_list=[scoring_function_component])
        step = 0
        start_time = time.time()
        condition = True
        while condition:
            agent, score = self._take_step(agent=agent, optimiser=optimiser, scoring_function_current=scoring_function,
                                           step=step)
            step += 1
            condition = self.check_conditions(score=score, step=step)

        time_passed = time.time() - start_time
        self._scoring_table.add_score_for_agent(ScoringTableEntryDTO(agent=agent, score=(self.max_num_iteration - step),
                                                                     scoring_function_components=
                                                                     scoring_function_component))
        self._logger.log_message(
            f"Finished training agent. Score achieved: {score}"f".Component: {scoring_function_component.get('name')}")
        self._logger.log_message(f"Ended at step: {step}, time elapsed: {time_passed}s.")