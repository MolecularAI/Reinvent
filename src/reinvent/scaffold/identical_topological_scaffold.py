from copy import deepcopy

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .scaffold_filters import ScaffoldFilter
from .scaffold_parameters import ScaffoldParameters
from ..scoring.score_summary import FinalSummary
from ..utils.smiles import convert_to_rdkit_smiles


class IdenticalTopologicalScaffold(ScaffoldFilter):
    """Penalizes compounds based on exact Topological Scaffolds previously generated."""

    def __init__(self, parameters: ScaffoldParameters):
        super().__init__(parameters)

    def score(self, score_summary: FinalSummary) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for i in score_summary.valid_idxs:
            smile = convert_to_rdkit_smiles(smiles[i])
            scaffold = self._calculate_scaffold(smile)
            scores[i] = 0 if self._smiles_exists(scaffold, smile) else scores[i]
            if scores[i] >= self.parameters.minscore:
                self._add_to_memory(i, scores[i], smile, scaffold, score_summary.scaffold_log)
                scores[i] = self._penalize_score(scaffold, scores[i])
        return scores

    def _calculate_scaffold(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            try:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
                scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            except ValueError:
                scaffold_smiles = ''
        else:
            scaffold_smiles = ''
        return scaffold_smiles
