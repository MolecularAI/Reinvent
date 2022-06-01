from dataclasses import dataclass
from typing import List

from reinvent_scoring import ScoringFunctionParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters

from running_modes.configurations import InceptionConfiguration
from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import \
    LearningStrategyConfiguration


@dataclass
class LinkInventProductionStrategyConfiguration:
    name: str
    input: List[str]
    learning_strategy: LearningStrategyConfiguration
    scoring_function: ScoringFunctionParameters
    diversity_filter: DiversityFilterParameters
    inception: InceptionConfiguration
    retain_inception: bool
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120
    number_of_steps: int = 100
    randomize_input: bool = False
