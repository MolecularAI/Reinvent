from dataclasses import dataclass

from reinvent_chemistry.library_design.reaction_filters.reaction_filter_configruation import ReactionFilterConfiguration
from reinvent_scoring import ScoringFuncionParameters
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters


@dataclass
class ScoringStrategyConfiguration:
    reaction_filter: ReactionFilterConfiguration
    diversity_filter: DiversityFilterParameters
    scoring_function: ScoringFuncionParameters
    name: str
