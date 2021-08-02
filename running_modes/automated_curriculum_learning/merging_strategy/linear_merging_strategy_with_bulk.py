import time
import torch
from typing import Tuple
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.merging_strategy.base_merging_strategy import BaseMergingStrategy
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations.automated_curriculum_learning.merging_strategy_configuration import MergingStrategyConfiguration


class LinearMergingStrategyWithBulk(BaseMergingStrategy):
    """ Train an agent by gradually increasing the difficulty of the scoring function."""

    def __init__(self, prior: Model, name: str, configuration: MergingStrategyConfiguration,
                 diversity_filter: BaseDiversityFilter, logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        super().__init__(prior=prior, name=name, configuration=configuration, diversity_filter=diversity_filter,
                         logger=logger, scoring_table=scoring_table)

        self.bulk_size = self._parameters.special_parameters.get(self._special_parameters_enum.BULK_SIZE)

    def _merge(self, number_components: int, step: int) -> BaseScoringFunction:
        sf_parameters_to_combine = self._scoring_table.get_top_sf_components(number=number_components)
        scoring_function = self.setup_scoring_function(name=self._sf_name, parameter_list=sf_parameters_to_combine,
                                                       step=step)
        return scoring_function

    def run(self) -> Tuple[Model, int]:
        agent = self._scoring_table.get_top_agent()
        optimiser = torch.optim.Adam(agent.network.parameters(), lr=self._parameters.lr)
        step_counter = 0
        
        for sf_component in range(self.bulk_size, len(self._scoring_table.scoring_table)+1):
            start_time = time.time()
            self.disable_prior_gradients()
            scoring_function = self._merge(number_components=sf_component, step=step_counter)
            self.save_and_flush_memory(agent=agent, memory_name=f"_merge_{sf_component}")
            self._logger.log_message(f"Number of Components:{sf_component} is on use")
            agent, step_counter = self.train_agent(agent=agent, optimiser=optimiser, scoring_function=scoring_function,
                                                   step_counter=step_counter, start_time=start_time)

        return agent, step_counter
