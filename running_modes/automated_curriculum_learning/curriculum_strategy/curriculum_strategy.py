from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter

from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.curriculum_strategy.user_defined_curriculum import \
    UserDefinedCurriculum
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.configurations import CurriculumStrategyConfiguration
from running_modes.enums.curriculum_strategy_enum import CurriculumStrategyEnum
from running_modes.reinforcement_learning.inception import Inception


class CurriculumStrategy:
    def __new__(cls, prior: Model, agent: Model, configuration: CurriculumStrategyConfiguration,
                logger: BaseAutoCLLogger) -> BaseCurriculumStrategy:
        curriculum_strategy_enum = CurriculumStrategyEnum()

        diversity_filter = DiversityFilter(configuration.diversity_filter)
        first_scoring_function_instance = ScoringFunctionFactory(configuration.curriculum_objectives[0].scoring_function)
        inception = Inception(configuration.inception, first_scoring_function_instance, prior)

        if curriculum_strategy_enum.USER_DEFINED == configuration.name:
            return UserDefinedCurriculum(prior=prior, agent=agent, configuration=configuration,
                                         diversity_filter=diversity_filter, logger=logger, inception=inception)
        else:
            raise NotImplementedError(f"Unknown curriculum strategy name {configuration.name}")
