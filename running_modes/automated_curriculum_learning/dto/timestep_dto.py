from dataclasses import dataclass

import torch
from reinvent_scoring.scoring import FinalSummary


@dataclass
class TimestepDTO:
    start_time: float
    n_steps: int
    step: int
    score_summary: FinalSummary
    agent_likelihood: torch.tensor
    prior_likelihood: torch.tensor
    augmented_likelihood: torch.tensor