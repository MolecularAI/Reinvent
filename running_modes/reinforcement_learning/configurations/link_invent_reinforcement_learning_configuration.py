from dataclasses import dataclass
from typing import List

from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.reinforcement_learning.configurations.link_invent_scoring_strategy_congfiguration import \
    LinkInventScoringStrategyConfiguration


@dataclass
class LinkInventReinforcementLearningConfiguration:
    actor: str
    critic: str
    warheads: List[str]
    learning_strategy: LearningStrategyConfiguration
    scoring_strategy: LinkInventScoringStrategyConfiguration
    n_steps: int = 1000
    learning_rate: float = 0.0001
    batch_size: int = 128
    randomize_warheads: bool = False

