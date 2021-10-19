from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveLearningRateEnum:
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"
    CONSTANT = "constant"
