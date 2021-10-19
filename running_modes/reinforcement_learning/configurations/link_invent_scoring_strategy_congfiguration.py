from dataclasses import dataclass
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters

from running_modes.reinforcement_learning.configurations.scoring_strategy_configuration import \
    ScoringStrategyConfiguration


@dataclass
class LinkInventScoringStrategyConfiguration(ScoringStrategyConfiguration):
    diversity_filter: DiversityFilterParameters

