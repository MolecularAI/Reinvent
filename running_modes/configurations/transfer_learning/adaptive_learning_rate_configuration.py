from dataclasses import dataclass


@dataclass
class AdaptiveLearningRateConfiguration:
    mode: str = "constant"
    gamma: float = 0.8
    step: int = 1
    start: float = 5E-4
    min: float = 1E-5
    threshold: float = 1E-4
    average_steps: int = 4
    patience: int = 8
    restart_value: float = 1E-5
    sample_size: int = 100
    restart_times: int = 0
