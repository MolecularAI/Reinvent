from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter

from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.curriculum_strategy.linkinvent_curriculum_strategy import \
    LinkInventCurriculumStrategy
from running_modes.automated_curriculum_learning.curriculum_strategy.no_curriculum_strategy import NoCurriculumStrategy
from running_modes.automated_curriculum_learning.curriculum_strategy.reinvent_curriculum_strategy import \
    ReinventCurriculumStrategy
from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.configurations.automated_curriculum_learning.curriculum_strategy_input_configuration import \
    CurriculumStrategyInputConfiguration
from running_modes.enums.curriculum_strategy_enum import CurriculumStrategyEnum


class CurriculumStrategy:
    def __new__(cls, prior: GenerativeModelBase, agent: GenerativeModelBase,
                configuration: CurriculumStrategyInputConfiguration,
                logger: BaseLogger) -> BaseCurriculumStrategy:

        curriculum_strategy_enum = CurriculumStrategyEnum()
        first_objective = configuration.curriculum_objectives[0]
        first_scoring_function_instance = ScoringFunctionFactory(first_objective.scoring_function)
        inception = Inception(configuration.inception, first_scoring_function_instance, prior)
        diversity_filter = DiversityFilter(configuration.diversity_filter)

        if curriculum_strategy_enum.USER_DEFINED == configuration.name:
            cl_strategy = ReinventCurriculumStrategy(prior=prior, agent=agent, configuration=configuration,
                                                     diversity_filter=diversity_filter, logger=logger, inception=inception)
            return cl_strategy
        elif curriculum_strategy_enum.LINK_INVENT == configuration.name:
            cl_strategy = LinkInventCurriculumStrategy(prior=prior, agent=agent, configuration=configuration,
                                                       diversity_filter=diversity_filter, logger=logger, inception=inception)
            return cl_strategy
        elif curriculum_strategy_enum.NO_CURRICULUM == configuration.name:
            cl_strategy = NoCurriculumStrategy(prior=prior, agent=agent, configuration=configuration,
                                                      diversity_filter=diversity_filter, logger=logger, inception=inception)
            return cl_strategy
        else:
            raise NotImplementedError(f"Unknown curriculum strategy name {configuration.name}")
