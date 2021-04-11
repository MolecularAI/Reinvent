from copy import deepcopy

import numpy as np

from diversity_filters.base_diversity_filter import BaseDiversityFilter
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from reinvent_scoring.scoring.score_summary import FinalSummary


class NoScaffoldFilter(BaseDiversityFilter):
    """Don't penalize compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles
        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.minscore:
                smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])
                self._add_to_memory(i, scores[i], smile, smile, score_summary.scaffold_log, step)
        return scores
