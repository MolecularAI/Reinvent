from dataclasses import dataclass


@dataclass
class CurriculumLearningConfiguration:
    prior: str
    agent: str
    update_lock: str
    general_configuration_path: str
    pause_lock: str = ''
    pause_limit: int = 5
    n_steps: int = 3000
    sigma: int = 120
    learning_rate: float = 0.0001
    batch_size: int = 128
    reset: int = 0
    reset_score_cutoff: float = 0.5
    margin_threshold: int = 50
    distance_threshold: float = -100.0
    scheduled_update_step: int = 0
