from dataclasses import dataclass


@dataclass
class ReinforcementLearningConfiguration:
    prior: str
    agent: str
    n_steps: int = 3000
    sigma: int = 120
    learning_rate: float = 0.0001
    batch_size: int = 128
    margin_threshold: int = 50
