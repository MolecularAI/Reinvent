from running_modes.configurations.automated_curriculum_learning.base_configuration import BaseConfiguration
from running_modes.configurations.automated_curriculum_learning.curriculum_strategy_input_configuration import \
    CurriculumStrategyInputConfiguration
from running_modes.configurations.automated_curriculum_learning.prodcution_strategy_input_configuration import \
    ProductionStrategyInputConfiguration


class AutomatedCurriculumLearningInputConfiguration(BaseConfiguration):
    agent: str
    prior: str
    curriculum_strategy: CurriculumStrategyInputConfiguration
    production_strategy: ProductionStrategyInputConfiguration
