from abc import abstractmethod

from rdkit import Chem
from typing import List
import numpy as np


class ScoringTest:

    @property
    @abstractmethod
    def component(self):
        raise NotImplementedError("No attribute 'component' in a ScoringTest. Perhaps missing 'cls.component = ...' in "
                                  "setUpClass?")

    def score(self, smile: str) -> float:
        mol = Chem.MolFromSmiles(smile)
        score = self.component.calculate_score([mol])
        return score.total_score[0]

    def multiple_scores(self, smiles: List[str]) -> np.array:
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        score = self.component.calculate_score(mols)
        return score.total_score
