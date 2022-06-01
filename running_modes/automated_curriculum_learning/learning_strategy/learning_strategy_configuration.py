from dataclasses import dataclass

from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_enum import LearningStrategyEnum


@dataclass
class LearningStrategyConfiguration:
    name: str = LearningStrategyEnum().DAP_SINGLE_QUERY
    parameters: dict = None
