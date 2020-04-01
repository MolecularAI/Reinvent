from dataclasses import dataclass


@dataclass
class AdaptiveLearningRateConfiguration:
    mode = "constant"
    gamma = 0.8
    step = 1
    start = 5E-4
    min = 1E-5
    threshold = 1E-4
    average_steps = 4
    patience = 8
    restart_value = 1E-5
    sample_size = 100
    restart_times = 0
