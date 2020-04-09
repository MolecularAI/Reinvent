import numpy as np
from rdkit.Chem.Descriptors import qed
from typing import List

from ..component_parameters import ComponentParameters
from .base_score_component import BaseScoreComponent
from ..score_summary import ComponentSummary


class QedScore(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._calculate_qed(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _calculate_qed(self, query_mols) -> np.array:
        qed_scores = []
        for mol in query_mols:
            try:
                qed_score = qed(mol)
            except ValueError:
                qed_score = 0.0
            qed_scores.append(qed_score)
        return np.array(qed_scores, dtype=np.float32)