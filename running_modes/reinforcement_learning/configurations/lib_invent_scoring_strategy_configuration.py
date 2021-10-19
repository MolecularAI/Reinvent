from dataclasses import dataclass

from reinvent_chemistry.library_design.reaction_filters.reaction_filter_configruation import ReactionFilterConfiguration
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters

from running_modes.reinforcement_learning.configurations.scoring_strategy_configuration import \
    ScoringStrategyConfiguration


@dataclass
class LibInventScoringStrategyConfiguration(ScoringStrategyConfiguration):
    reaction_filter: ReactionFilterConfiguration
    diversity_filter: DiversityFilterParameters
