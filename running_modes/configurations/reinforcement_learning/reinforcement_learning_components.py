from dataclasses import dataclass

from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.configurations.reinforcement_learning.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration


@dataclass
class ReinforcementLearningComponents:
    """This class holds the necessary configuration components to run RL"""
    reinforcement_learning: ReinforcementLearningConfiguration
    scoring_function: ScoringFunctionParameters
    diversity_filter: DiversityFilterParameters
    inception: InceptionConfiguration
