from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter

from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.automated_curriculum_learning.production_strategy.link_invent_production_strategy import \
    LinkInventProductionStrategy
from running_modes.automated_curriculum_learning.production_strategy.base_production_strategy import \
    BaseProductionStrategy
from running_modes.automated_curriculum_learning.production_strategy.reinvent_production_strategy import \
    ReinventProductionStrategy
from running_modes.configurations import ProductionStrategyInputConfiguration
from running_modes.enums.production_strategy_enum import ProductionStrategyEnum


class ProductionStrategy:
    def __new__(cls, prior: GenerativeModelBase, inception: Inception,
                configuration: ProductionStrategyInputConfiguration,
                logger: BaseLogger) -> BaseProductionStrategy:

        production_strategy_enum = ProductionStrategyEnum()
        scoring_function_instance = ScoringFunctionFactory(configuration.scoring_function)
        diversity_filter = DiversityFilter(configuration.diversity_filter)

        if production_strategy_enum.STANDARD == configuration.name:
             production = ReinventProductionStrategy(prior=prior,
                                              diversity_filter=diversity_filter, inception=inception,
                                              scoring_function=scoring_function_instance,
                                              configuration=configuration, logger=logger)
             return production
        elif production_strategy_enum.LINK_INVENT == configuration.name:
            production = LinkInventProductionStrategy(prior=prior, diversity_filter=diversity_filter,
                                                      inception=inception,
                                                      scoring_function=scoring_function_instance,
                                                      configuration=configuration, logger=logger)
            return production
        else:
            raise NotImplementedError(f"Unknown production strategy {configuration.name}")
