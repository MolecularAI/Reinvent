from dataclasses import dataclass
from typing import List

from running_modes.lib_invent.configurations.learning_strategy_configuration import LearningStrategyConfiguration
from running_modes.lib_invent.configurations.scoring_strategy_configuration import ScoringStrategyConfiguration


@dataclass
class ReinforcementLearningConfiguration:
    actor: str
    critic: str
    scaffolds: List[str]
    learning_strategy: LearningStrategyConfiguration
    scoring_strategy: ScoringStrategyConfiguration
    n_steps: int = 1000
    learning_rate: float = 0.0001
    batch_size: int = 128
    randomize_scaffolds: bool = False
