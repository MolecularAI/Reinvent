import time
import torch
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction
from typing import Tuple
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.merging_strategy.base_merging_strategy import BaseMergingStrategy
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations.automated_curriculum_learning.merging_strategy_configuration import \
    MergingStrategyConfiguration

from running_modes.reinforcement_learning.inception import Inception


class LinearMergingStrategyWithThreshold(BaseMergingStrategy):
    def __init__(self, prior: Model, scoring_function_name: str, diversity_filter: BaseDiversityFilter,
                 inception: Inception, configuration: MergingStrategyConfiguration,
                 logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        super().__init__(prior=prior, scoring_function_name=scoring_function_name, configuration=configuration,
                         diversity_filter=diversity_filter, inception=inception, logger=logger, scoring_table=scoring_table)

    def _merge(self, number_components: int, step: int) -> BaseScoringFunction:
        sf_parameters_to_combine = self._scoring_table.get_top_sf_components(number=number_components)
        scoring_function = self.setup_scoring_function(name=self._scoring_function_name, parameter_list=sf_parameters_to_combine, step=step)
        return scoring_function

    def run(self) -> Tuple[Model, int]:
        agent = self._scoring_table.get_top_agent()
        optimiser = torch.optim.Adam(agent.network.parameters(), lr=self._parameters.learning_rate)
        step_counter = 0
        num_components = len(self._scoring_table.scoring_table)

        for sf_component in range(num_components):
            final_component = (num_components - sf_component) == 1
            start_time = time.time()
            self.disable_prior_gradients()
            scoring_function = self._merge(number_components=sf_component + 1, step=step_counter)
            self.save_and_flush_memory(agent=agent, memory_name=f"_merge_{sf_component}")
            self._logger.log_message(f"Component number:{sf_component + 1} is on use")
            agent, step_counter = self.train_agent(agent=agent, optimiser=optimiser, scoring_function=scoring_function,
                                                   step_counter=step_counter, start_time=start_time, final_component=final_component)

        return agent, step_counter
