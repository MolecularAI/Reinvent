from dataclasses import dataclass

from running_modes.configurations.automated_curriculum_learning.strategy_parameters_configuration import \
    RankingStrategyParametersConfiguration


@dataclass
class RankingStrategyConfiguration:
    name: str
    parameters: RankingStrategyParametersConfiguration