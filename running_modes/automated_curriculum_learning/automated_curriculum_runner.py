from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFuncionParameters, ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.merging_strategy.merging_strategy import MergingStrategy
from running_modes.automated_curriculum_learning.production_strategy.production_strategy import ProductionStrategy
from running_modes.automated_curriculum_learning.ranking_strategy.ranking_strategy import RankingStrategy
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations import AutomatedCurriculumLearningComponents, InceptionConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.inception import Inception


class AutomatedCurriculumRunner(BaseRunningMode):
    def __init__(self, config: AutomatedCurriculumLearningComponents,
                 logger: BaseAutoCLLogger, prior: Model, agent: Model):
        self._scoring_table = ScoringTable()

        self._config = config.automated_curriculum_learning
        self._prior = prior
        self._agent = agent
        self._logger = logger

        # Ranking + Merging Phases SF, DF, Inception
        merging_sf_params = ScoringFuncionParameters(**self._config.merging_strategy.scoring_function)
        merging_sf = self._setup_scoring_function(self._config.merging_strategy.scoring_function)
        merging_df = self._setup_diversity_filter(self._config.merging_strategy.diversity_filter)
        merging_inception = self._setup_inception(self._config.merging_strategy.inception, merging_sf)

        # Production Phase SF, DF
        production_sf_params = ScoringFuncionParameters(**self._config.production_strategy.scoring_function)
        production_sf = self._setup_scoring_function(self._config.production_strategy.scoring_function)
        production_df = self._setup_diversity_filter(self._config.production_strategy.diversity_filter)

        self.ranking_strategy = RankingStrategy(prior=self._prior, agent=self._agent,
                                                scoring_function_params=merging_sf_params,
                                                diversity_filter=merging_df, inception=merging_inception,
                                                configuration=self._config.ranking_strategy,
                                                logger=self._logger, scoring_table=self._scoring_table)

        self.merging_strategy = MergingStrategy(prior=self._prior, scoring_function_name=merging_sf_params.name,
                                                diversity_filter=merging_df, inception=merging_inception,
                                                configuration=self._config.merging_strategy,
                                                logger=self._logger, scoring_table=self._scoring_table)

        # check if user specifies to retain the Merging Phase Inception
        # if not, the Production Phase Inception is initialized and all previous Merging Inception memory is purged
        if self._config.production_strategy.retain_inception:
            production_inception = merging_inception
        else:
            production_inception = self._setup_inception(self._config.production_strategy.inception, production_sf)

        self.production_strategy = ProductionStrategy(prior=self._prior, scoring_function_params=production_sf_params,
                                                      diversity_filter=production_df, inception=production_inception,
                                                      configuration=self._config.production_strategy,
                                                      logger=self._logger, scoring_table=self._scoring_table)

    def run(self):
        # Ranking
        self._logger.log_text_to_file("*** Ranking begins ***")
        self._logger.log_message("Ranking")
        self.ranking_strategy.rank_agents()
        # Merging
        self._logger.log_message("Merging")
        self._logger.log_text_to_file("*** Merging begins ***")
        cl_agent, step_counter = self.merging_strategy.run()
        # Production
        self._logger.log_message("Production")
        self._logger.log_text_to_file("*** Production begins ***")
        self.production_strategy.run(cl_agent, step_counter)

    def _setup_scoring_function(self, scoring_function_parameters):
        scoring_function_parameters = ScoringFuncionParameters(**scoring_function_parameters)
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        return scoring_function_instance

    def _setup_diversity_filter(self, diversity_filter_parameters):
        diversity_filter_parameters = DiversityFilterParameters(**diversity_filter_parameters)
        diversity_filter = DiversityFilter(diversity_filter_parameters)
        return diversity_filter

    def _setup_inception(self, inception_parameters, scoring_function):
        # "scoring_function" is usually a class variable but change it here to a regular variable
        # to give the flexibility of setting up Inception using the merging/production scoring function
        inception_config = InceptionConfiguration(**inception_parameters)
        inception = Inception(inception_config, scoring_function, self._prior)
        return inception