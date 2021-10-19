from dataclasses import dataclass

from reinvent_scoring import ScoringFunctionParameters
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters

from running_modes.configurations import InceptionConfiguration


@dataclass
class ProductionStrategyConfiguration:
    name: str
    scoring_function: ScoringFunctionParameters
    diversity_filter: DiversityFilterParameters
    inception: InceptionConfiguration
    retain_inception: bool
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120
    n_steps: int = 100
