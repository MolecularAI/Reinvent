from typing import List

import numpy as np
from rdkit import Chem

from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary


class MatchingSubstructure(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.target_smarts = self.parameters.smiles  # these are actually smarts
        self._validate_inputs(self.parameters.smiles)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._substructure_match(molecules, self.target_smarts)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _smiles_to_fingerprints(self, smiles: List[str], radius=3, useCounts=True, useFeatures=True) -> []:
        # This is intentionally doing nothing as the input is expected to be in smarts rather than in smiles
        idx = []
        fps = []
        return fps, idx

    def _substructure_match(self, query_mols, list_of_SMARTS):
        if len(list_of_SMARTS) == 0:
            return np.ones(len(query_mols), dtype=np.float32)

        match = [any([mol.HasSubstructMatch(Chem.MolFromSmarts(subst)) for subst in list_of_SMARTS
                      if Chem.MolFromSmarts(subst)]) for mol in query_mols]
        return 0.5 * (1 + np.array(match))

    def _validate_inputs(self, smiles):
        for smart in smiles:
            if Chem.MolFromSmarts(smart) is None:
                raise IOError(f"Invalid smarts pattern provided as a matching substructure: {smart}")
