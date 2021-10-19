from dataclasses import dataclass


@dataclass
class LinkInventLearningRateConfiguration:
    start: float = 0.0001
    min: float = 0.000001
    gamma: float = 0.95
    step: int = 1
