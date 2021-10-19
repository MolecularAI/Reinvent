from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter

from running_modes.automated_curriculum_learning.curriculum_strategy.curriculum_strategy import CurriculumStrategy
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.production_strategy.production_strategy import ProductionStrategy
from running_modes.configurations import AutomatedCLConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.inception import Inception


class AutomatedCurriculumRunner(BaseRunningMode):
    def __init__(self, config: AutomatedCLConfiguration, logger: BaseAutoCLLogger, prior: Model, agent: Model):

        self._config = config
        self._prior = prior
        self._agent = agent
        self._logger = logger
        self._curriculum_strategy = CurriculumStrategy(self._prior, self._agent, self._config.curriculum_strategy,
                                                       self._logger)

        production_sf = ScoringFunctionFactory(self._config.production_strategy.scoring_function)
        production_df = DiversityFilter(self._config.production_strategy.diversity_filter)

        if self._config.production_strategy.retain_inception:
            production_inception = self._curriculum_strategy.inception
        else:
            production_inception = Inception(self._config.production_strategy.inception, production_sf, self._prior)

        self.production_strategy = ProductionStrategy(prior=self._prior,
                                                      diversity_filter=production_df, inception=production_inception,
                                                      configuration=self._config.production_strategy,
                                                      logger=self._logger)

    def run(self):
        self._logger.log_message("Starting Curriculum Learning")
        cl_agent, step_counter = self._curriculum_strategy.run()
        # Production
        self._logger.log_message("Starting Production phase")
        self.production_strategy.run(cl_agent, step_counter)
