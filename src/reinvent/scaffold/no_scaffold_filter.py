from copy import deepcopy

import numpy as np

from .scaffold_filters import ScaffoldFilter
from .scaffold_parameters import ScaffoldParameters
from ..scoring.score_summary import FinalSummary
from ..utils.smiles import convert_to_rdkit_smiles


class NoScaffoldFilter(ScaffoldFilter):
    """Don't penalize compounds."""

    def __init__(self, parameters: ScaffoldParameters):
        super().__init__(parameters)

    def score(self, score_summary: FinalSummary) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles
        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.minscore:
                smile = convert_to_rdkit_smiles(smiles[i])
                self._add_to_memory(i, scores[i], smile, smile, score_summary.scaffold_log)
        return scores
