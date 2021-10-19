from dataclasses import dataclass
from typing import List

from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.reinforcement_learning.configurations.lib_invent_scoring_strategy_configuration import \
    LibInventScoringStrategyConfiguration


@dataclass
class LibInventReinforcementLearningConfiguration:
    actor: str
    critic: str
    scaffolds: List[str]
    learning_strategy: LearningStrategyConfiguration
    scoring_strategy: LibInventScoringStrategyConfiguration
    n_steps: int = 1000
    learning_rate: float = 0.0001
    batch_size: int = 128
    randomize_scaffolds: bool = False
