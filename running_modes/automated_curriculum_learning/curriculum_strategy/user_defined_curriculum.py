import time
from typing import Tuple

import torch
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter

from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.configurations import CurriculumStrategyConfiguration
from running_modes.reinforcement_learning.inception import Inception


class UserDefinedCurriculum(BaseCurriculumStrategy):
    def __init__(self, prior: Model, agent: Model, diversity_filter: BaseDiversityFilter,
                 inception: Inception, configuration: CurriculumStrategyConfiguration, logger: BaseAutoCLLogger):
        super().__init__(prior=prior, agent=agent, configuration=configuration, diversity_filter=diversity_filter,
                         inception=inception, logger=logger)

    def run(self) -> Tuple[Model, int]:
        optimiser = torch.optim.Adam(self._agent.network.parameters(), lr=self._parameters.learning_rate)
        step_counter = 0
        self.disable_prior_gradients()

        for item_id, sf_configuration in enumerate(self._parameters.curriculum_objectives):
            start_time = time.time()
            scoring_function = self._setup_scoring_function(item_id)
            step_counter = self.promote_agent(agent=self._agent, optimiser=optimiser, scoring_function=scoring_function,
                                              step_counter=step_counter, start_time=start_time,
                                              merging_threshold=sf_configuration.score_threshold)
            self.save_and_flush_memory(agent=self._agent, memory_name=f"_merge_{item_id}")

        return self._agent, step_counter
