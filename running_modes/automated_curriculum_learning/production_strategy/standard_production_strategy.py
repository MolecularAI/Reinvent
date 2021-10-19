import time

import torch
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.production_strategy.base_production_strategy import \
    BaseProductionStrategy
from running_modes.configurations.automated_curriculum_learning.production_strategy_configuration import \
    ProductionStrategyConfiguration
from running_modes.reinforcement_learning.inception import Inception


class StandardProductionStrategy(BaseProductionStrategy):
    def __init__(self, prior: Model,
                 diversity_filter: BaseDiversityFilter, inception: Inception, scoring_function: BaseScoringFunction,
                 configuration: ProductionStrategyConfiguration,
                 logger: BaseAutoCLLogger):
        super().__init__(prior=prior,
                         diversity_filter=diversity_filter, inception=inception, scoring_function=scoring_function,
                         configuration=configuration,
                         logger=logger)

    def run(self, cl_agent: Model, steps_so_far: int):
        # self._diversity_filter.flush_memory()
        start_time = time.time()
        self._disable_prior_gradients()
        optimizer = torch.optim.Adam(cl_agent.network.parameters(), lr=self._parameters.learning_rate)

        step_limit = steps_so_far + self._parameters.n_steps

        for step in range(steps_so_far, step_limit):
            self._take_step(agent=cl_agent, optimizer=optimizer, scoring_function_current=self._scoring_function,
                            step=step, start_time=start_time)

        self._logger.log_message(f"Production finished at step {step_limit}")
        self._logger.save_final_state(cl_agent, self._diversity_filter)
        self._logger.log_out_input_configuration()
