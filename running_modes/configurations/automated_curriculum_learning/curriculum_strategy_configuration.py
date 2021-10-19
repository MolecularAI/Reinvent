from dataclasses import dataclass
from typing import List

from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters

from running_modes.configurations import InceptionConfiguration
from running_modes.configurations.automated_curriculum_learning.curriculum_objective import CurriculumObjective


@dataclass
class CurriculumStrategyConfiguration:
    name: str
    curriculum_objectives: List[CurriculumObjective]
    diversity_filter: DiversityFilterParameters
    inception: InceptionConfiguration
    max_num_iterations: int
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120
