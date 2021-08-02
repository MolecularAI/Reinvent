from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFuncionParameters
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.production_strategy.base_production_strategy import \
    BaseProductionStrategy
from running_modes.automated_curriculum_learning.production_strategy.standard_production_strategy import \
    StandardProductionStrategy
from running_modes.automated_curriculum_learning.production_strategy.production_strategy_over_specific_components import \
    ProductionStrategyOverSpecificComponent
from running_modes.configurations.automated_curriculum_learning.production_strategy_configuration import \
    ProductionStrategyConfiguration
from running_modes.enums.production_strategy_enum import ProductionStrategyEnum
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.reinforcement_learning.inception import Inception


class ProductionStrategy:
    def __new__(cls, prior: Model, scoring_function_params: ScoringFuncionParameters,
                diversity_filter: BaseDiversityFilter, inception: Inception, configuration: ProductionStrategyConfiguration,
                logger: BaseAutoCLLogger, scoring_table: ScoringTable) -> BaseProductionStrategy:

        production_strategy_enum = ProductionStrategyEnum()

        if production_strategy_enum.STANDARD == configuration.name:
            return StandardProductionStrategy(prior=prior, scoring_function_params=scoring_function_params,
                                              diversity_filter=diversity_filter, inception=inception,
                                              configuration=configuration, logger=logger, scoring_table=scoring_table)

        elif production_strategy_enum.SPECIFIC_COMPONENTS == configuration.name:
            return ProductionStrategyOverSpecificComponent(prior=prior, scoring_function_params=scoring_function_params,
                                                           diversity_filter=diversity_filter, inception=inception,
                                                           configuration=configuration, logger=logger, scoring_table=scoring_table)

        else:
            raise NotImplementedError(f"Unknown production strategy {configuration.name}")
