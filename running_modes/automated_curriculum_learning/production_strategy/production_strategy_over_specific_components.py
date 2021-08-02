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
from running_modes.enums.special_parameters_enum import SpecialParametersEnum
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable


class ProductionStrategyOverSpecificComponent(BaseProductionStrategy):
    def __init__(self, prior: Model, scoring_function_params: ScoringFuncionParameters,
                 configuration: ProductionStrategyConfiguration, diversity_filter: BaseDiversityFilter,
                 logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        super().__init__(prior=prior, configuration=configuration,
                         scoring_function_params=scoring_function_params, diversity_filter=diversity_filter,
                         logger=logger, scoring_table=scoring_table)

        self._special_parameters_enum = SpecialParametersEnum()

    def run(self, cl_agent: Model, steps_so_far: int):
        self._diversity_filter.flush_memory()
        selected_components = self._parameters.special_parameters.get(self._special_parameters_enum.SELECTED_COMPONENTS)
        sf_parameters = self._scoring_table.get_sf_components_by_name(names=selected_components)
        scoring_function = self.setup_scoring_function(name=self.sf_name, parameter_list=sf_parameters)
        start_time = time.time()
        self._disable_prior_gradients()
        optimizer = torch.optim.Adam(cl_agent.network.parameters(), lr=self._parameters.lr)

        for step in range(steps_so_far, steps_so_far + self._parameters.n_steps):
            self._take_step(agent=cl_agent, optimizer=optimizer, scoring_function_current=scoring_function,
                            step=step, start_time=start_time)

        self._logger.log_text_to_file(f"Production finished at step {step}")
        self._logger.save_final_state(cl_agent, self._diversity_filter)
        self._logger.log_out_input_configuration()
