from dataclasses import dataclass

from running_modes.configurations.automated_curriculum_learning.strategy_parameters_configuration import \
    ProductionStrategyParametersConfiguration


@dataclass
class ProductionStrategyConfiguration:
    # production strategy name
    name: str
    # production sf, df, inception shared with ranking
    scoring_function: dict
    diversity_filter: dict
    inception: dict
    # boolean to denote whether to retain the merging phase inception memory
    retain_inception: bool
    parameters: ProductionStrategyParametersConfiguration
