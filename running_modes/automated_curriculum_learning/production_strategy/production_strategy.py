from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.production_strategy.base_production_strategy import \
    BaseProductionStrategy
from running_modes.automated_curriculum_learning.production_strategy.standard_production_strategy import \
    StandardProductionStrategy
from running_modes.configurations.automated_curriculum_learning.production_strategy_configuration import \
    ProductionStrategyConfiguration
from running_modes.enums.production_strategy_enum import ProductionStrategyEnum
from running_modes.reinforcement_learning.inception import Inception


class ProductionStrategy:
    def __new__(cls, prior: Model, diversity_filter: BaseDiversityFilter, inception: Inception,
                configuration: ProductionStrategyConfiguration, logger: BaseAutoCLLogger) -> BaseProductionStrategy:

        production_strategy_enum = ProductionStrategyEnum()
        scoring_function_instance = ScoringFunctionFactory(configuration.scoring_function)

        if production_strategy_enum.STANDARD == configuration.name:
            return StandardProductionStrategy(prior=prior,
                                              diversity_filter=diversity_filter, inception=inception,
                                              scoring_function=scoring_function_instance,
                                              configuration=configuration, logger=logger)

        else:
            raise NotImplementedError(f"Unknown production strategy {configuration.name}")
