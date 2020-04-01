from typing import List

import numpy as np

from scoring.component_parameters import ComponentParameters
from scoring.function.base_scoring_function import BaseScoringFunction
from scoring.score_summary import ComponentSummary


class CustomSum(BaseScoringFunction):

    def __init__(self, parameters: List[ComponentParameters], parallel=False):
        super().__init__(parameters, parallel)

    def _compute_non_penalty_components(self, summaries: List[ComponentSummary], smiles: List[str]):
        total_sum = np.full(len(smiles), 0, dtype=np.float32)
        all_weights = 0.

        for summary in summaries:
            if not self._component_is_penalty(summary):
                total_sum = total_sum + summary.total_score * summary.parameters.weight
                all_weights += summary.parameters.weight

        if all_weights == 0:
            """There are no non-penalty components and return array of ones. 
            This is needed so that it can work in cases where only penalty components are used"""
            return np.full(len(smiles), 1, dtype=np.float32)

        return total_sum / all_weights
