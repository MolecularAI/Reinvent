from dataclasses import dataclass
from typing import List

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters

from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.configurations import InceptionConfiguration
from running_modes.configurations.automated_curriculum_learning.curriculum_objective import CurriculumObjective



@dataclass
class LinkInventCurriculumStrategyConfiguration:
    name: str
    input: List[str]
    learning_strategy: LearningStrategyConfiguration
    curriculum_objectives: List[CurriculumObjective]
    diversity_filter: DiversityFilterParameters
    inception: InceptionConfiguration
    max_num_iterations: int
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120
    randomize_input: bool = False
