from abc import ABC, abstractmethod
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem

from ..component_parameters import ComponentParameters
from ..score_summary import ComponentSummary
from ...utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum


class BaseScoreComponent(ABC):

    def __init__(self, parameters: ComponentParameters):
        self.component_specific_parameters = ComponentSpecificParametersEnum()
        self.parameters = parameters

    @abstractmethod
    def calculate_score(self, molecules: List) -> ComponentSummary:
        raise NotImplementedError("calculate_score method is not implemented")

    def _smiles_to_fingerprints(self, smiles: List[str], radius=3, useCounts=True, useFeatures=True) -> []:
        mols, idx = self._smiles_to_mols(smiles)
        fingerprints = [AllChem.GetMorganFingerprint(
            mol,
            radius,
            useCounts=useCounts,
            useFeatures=useFeatures
        ) for mol in mols]
        return fingerprints, idx

    def _smiles_to_mols(self, query_smiles: List[str]) -> []:
        mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs

