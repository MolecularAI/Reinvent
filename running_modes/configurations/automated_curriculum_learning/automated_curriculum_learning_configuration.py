from dataclasses import dataclass

from running_modes.configurations.automated_curriculum_learning.merging_strategy_configuration import \
    MergingStrategyConfiguration
from running_modes.configurations.automated_curriculum_learning.production_strategy_configuration import \
    ProductionStrategyConfiguration
from running_modes.configurations.automated_curriculum_learning.ranking_strategy_configuration import \
    RankingStrategyConfiguration


@dataclass
class AutomatedCurriculumLearningConfiguration:
    ranking_strategy: RankingStrategyConfiguration
    merging_strategy: MergingStrategyConfiguration
    production_strategy: ProductionStrategyConfiguration


@dataclass
class AutomatedCurriculumLearningComponents:
    """This class holds the necessary configuration components to run Auto CL"""
    prior: str
    agent: str
    automated_curriculum_learning: AutomatedCurriculumLearningConfiguration