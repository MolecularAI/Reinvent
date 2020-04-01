from dataclasses import dataclass

import numpy as np
from typing import List

from scoring.component_parameters import ComponentParameters


@dataclass
class ComponentSummary:
    total_score: np.array
    parameters: ComponentParameters


class FinalSummary:
    def __init__(self, total_score: np.array, scored_smiles: List[str], valid_idxs: List[int],
                 scaffold_log_summary: List[ComponentSummary], log_summary: List[ComponentSummary]):
        self.total_score = total_score
        self.scored_smiles = scored_smiles
        self.valid_idxs = valid_idxs
        self.scaffold_log: List[ComponentSummary] = scaffold_log_summary
        self.profile: List[LoggableComponent] = \
            [LoggableComponent(c.parameters.component_type, c.parameters.name, c.total_score) for c
             in log_summary]


@dataclass
class LoggableComponent:
    component_type: str
    name: str
    score: np.array
