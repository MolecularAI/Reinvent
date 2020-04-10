import math
from typing import List

import numpy as np

from ..component_parameters import ComponentParameters
from .base_scoring_function import BaseScoringFunction
from ...scoring.score_summary import ComponentSummary


class CustomProduct(BaseScoringFunction):

    def __init__(self, parameters: List[ComponentParameters], parallel=False):
        super().__init__(parameters, parallel)

    def _calculate_pow(self, values, weight):
        y = [math.pow(value, weight) for value in values]
        return np.array(y, dtype=np.float32)

    def _get_all_weights(self, summaries: List[ComponentSummary]) -> int:
        all_weights = []

        for summary in summaries:
            if not self._component_is_penalty(summary):
                all_weights.append(summary.parameters.weight)
        return sum(all_weights)

    def _compute_non_penalty_components(self, summaries: List[ComponentSummary], smiles: List[str]):
        product = np.full(len(smiles), 1, dtype=np.float32)
        all_weights = self._get_all_weights(summaries)

        for summary in summaries:
            if not self._component_is_penalty(summary):
                comp_pow = self._calculate_pow(summary.total_score, summary.parameters.weight / all_weights)
                product = product * comp_pow

        return product
