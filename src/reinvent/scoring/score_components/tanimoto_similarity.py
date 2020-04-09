import numpy as np
from rdkit.Chem import DataStructs
from typing import List
from rdkit.Chem import AllChem

from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary


class TanimotoSimilarity(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._fingerprints, _ = self._smiles_to_fingerprints(self.parameters.smiles)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        query_fps = self._mols_to_fingerprints(molecules)
        score = self._calculate_tanimoto(query_fps, self._fingerprints)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _mols_to_fingerprints(self, molecules: List, radius=3, useCounts=True, useFeatures=True) -> []:
        fingerprints = [AllChem.GetMorganFingerprint(
            mol,
            radius,
            useCounts=useCounts,
            useFeatures=useFeatures
        ) for mol in molecules]
        return fingerprints

    def _calculate_tanimoto(self, query_fps, ref_fingerprints) -> np.array:
        return np.array([np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fingerprints)) for fp in query_fps])

