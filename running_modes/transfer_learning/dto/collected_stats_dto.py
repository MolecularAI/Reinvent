from dataclasses import dataclass
from typing import List

from running_modes.transfer_learning.dto.sampled_stats_dto import SampledStatsDTO


@dataclass
class CollectedStatsDTO:
    jsd_binned: float
    jsd_un_binned: float
    nll: List[float]
    training_stats: SampledStatsDTO
    validation_nll: List[float] = None
    validation_stats: SampledStatsDTO = None
