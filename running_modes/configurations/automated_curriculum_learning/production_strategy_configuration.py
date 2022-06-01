from pydantic import BaseModel
from reinvent_scoring import ScoringFunctionParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters

from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.configurations import InceptionConfiguration


class ProductionStrategyConfiguration(BaseModel):
    name: str
    learning_strategy: LearningStrategyConfiguration
    scoring_function: ScoringFunctionParameters
    diversity_filter: DiversityFilterParameters
    inception: InceptionConfiguration
    retain_inception: bool
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120
    number_of_steps: int = 100
