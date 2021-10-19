from dataclasses import dataclass
from typing import List


@dataclass
class SampledStatsDTO:
    nll_input_sampled_target: List[float]
    molecule_smiles: List[str]
    molecule_parts_smiles: List[str]
    valid_fraction: float
