from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class SampledBatchDTO:
    sequences: torch.Tensor
    smiles: List[str]
    likelihoods: torch.Tensor
