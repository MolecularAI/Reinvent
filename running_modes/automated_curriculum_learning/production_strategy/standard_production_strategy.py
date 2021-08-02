import time
import torch
from reinvent_scoring import ScoringFuncionParameters
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.production_strategy.base_production_strategy import \
    BaseProductionStrategy
from running_modes.configurations.automated_curriculum_learning.production_strategy_configuration import \
    ProductionStrategyConfiguration
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable

from running_modes.reinforcement_learning.inception import Inception


class StandardProductionStrategy(BaseProductionStrategy):
    def __init__(self, prior: Model, scoring_function_params: ScoringFuncionParameters,
                 diversity_filter: BaseDiversityFilter, inception: Inception, configuration: ProductionStrategyConfiguration,
                 logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        super().__init__(prior=prior, scoring_function_params=scoring_function_params,
                         diversity_filter=diversity_filter, inception=inception, configuration=configuration,
                         logger=logger, scoring_table=scoring_table)

        self._scoring_function = self.setup_scoring_function(name=self._scoring_function_name, parameter_list=self._scoring_function_params)

    def run(self, cl_agent: Model, steps_so_far: int):
        # self._diversity_filter.flush_memory()
        start_time = time.time()
        self._disable_prior_gradients()
        optimizer = torch.optim.Adam(cl_agent.network.parameters(), lr=self._parameters.learning_rate)

        for step in range(steps_so_far, steps_so_far + self._parameters.n_steps):
            self._take_step(agent=cl_agent, optimizer=optimizer, scoring_function_current=self._scoring_function,
                            step=step, start_time=start_time)

        self._logger.log_text_to_file(f"Production finished at step {step}")
        self._logger.save_final_state(cl_agent, self._diversity_filter)
        self._logger.log_out_input_configuration()
