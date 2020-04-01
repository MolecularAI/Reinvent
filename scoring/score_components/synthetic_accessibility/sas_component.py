import pickle
from typing import List, Tuple

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors

from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_components.synthetic_accessibility.sascorer import calculateScore
from scoring.score_summary import ComponentSummary


class SASComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.activity_model = self._load_model(parameters)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self.predict_from_molecules(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def predict_from_molecules(self, molecules: List) -> np.array:
        if len(molecules) == 0:
            return np.array([])

        descriptors = self._calculate_descriptors(molecules)
        sas_predictions = self.activity_model.predict_proba(descriptors)

        return sas_predictions[:, 1]

    def _load_model(self, parameters: ComponentParameters):
        try:
            activity_model = self._load_scikit_model(parameters.model_path)
        except:
            raise Exception(f"The loaded file {parameters.model_path} isn't a valid scikit-learn model")
        return activity_model

    def _load_scikit_model(self, model_path: str):
        with open(model_path, "rb") as f:
            scikit_model = pickle.load(f)
        return scikit_model

    def _calculate_descriptors(self, molecules: List) -> List:
        fingerprints = self._mols_to_fingerprint(molecules)
        descriptors = []

        for idx, mol in enumerate(molecules):
            others = np.array([calculateScore(mol), Descriptors.ExactMolWt(mol)])
            prop_array = np.concatenate([others, fingerprints[idx]]).reshape((1, -1))[0]
            descriptors.append(prop_array)
        return descriptors

    def _mols_to_fingerprint(self, mols) -> List:
        fingerprints = [AllChem.GetHashedMorganFingerprint(mol, 3, nBits=4096) for mol in mols]
        fp_array = []

        for fp in fingerprints:
            numpy_fingreprint = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, numpy_fingreprint)
            fp_array.append(numpy_fingreprint)

        return fp_array

    def _get_props(self, mol):
        molwt = Descriptors.ExactMolWt(mol)

        return molwt

    def _predict_sas(self, smiles: List[str], parameters: dict) -> Tuple[np.array, List]:
        fps, valid_idx = self._smiles_to_fingerprints(smiles, parameters)

        if len(valid_idx) == 0:
            return np.array([]), valid_idx
        activity = self.activity_model.predict_proba(fps, parameters)
        return activity, valid_idx
