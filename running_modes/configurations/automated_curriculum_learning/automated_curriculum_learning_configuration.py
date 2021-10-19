from dataclasses import dataclass

from running_modes.configurations.automated_curriculum_learning.base_configuration import BaseConfiguration
from running_modes.configurations.automated_curriculum_learning.curriculum_strategy_configuration import \
    CurriculumStrategyConfiguration
from running_modes.configurations.automated_curriculum_learning.production_strategy_configuration import \
    ProductionStrategyConfiguration


@dataclass
class AutomatedCLConfiguration(BaseConfiguration):
    prior: str
    agent: str
    curriculum_strategy: CurriculumStrategyConfiguration
    production_strategy: ProductionStrategyConfiguration
