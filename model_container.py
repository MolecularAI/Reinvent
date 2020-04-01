from typing import List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools

from scoring.score_transformations import TransformationFactory


class ModelContainer():
    def __init__(self, activity_model, model_type: str, specific_parameters: {}):
        """
        :type activity_model: scikit-learn type of model object
        :type model_type: can be "classification" or "regression"
        """
        self.__activity_model = activity_model
        self.__model_type = model_type
        self.__CONTAINER_VERSION = "2"
        self.transformation = self._assign_transformation(specific_parameters)
        self._molecules_to_descriptors = self._load_descriptor(specific_parameters)

    @property
    def activity_model(self):
        return self.__activity_model

    @activity_model.setter
    def activity_model(self, value):
        self.__activity_model = value

    @property
    def model_type(self):
        return self.__model_type

    @model_type.setter
    def model_type(self, value):
        self.__model_type = value

    def _smiles_to_fingerprints(self, molecules: List, parameters: {}) -> []:
        radius = parameters.get('radius', 3)
        size = parameters.get('size', 2048)
        fingerprints = []
        fp_bits = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, size) for mol in molecules]
        for fp in fp_bits:
            fp_np = np.zeros((1, size), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints

    def _smiles_to_count_fingerprints(self, molecules: List, parameters: {}) -> []:
        radius = parameters.get('radius', 3)
        useCounts = parameters.get('use_counts', True)
        useFeatures = parameters.get('use_features', True)
        size = parameters.get('size', 2048)
        fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures) for mol in molecules]
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp

    def _smiles_to_mols(self, query_smiles: List[str]) -> []:
        mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs

    def predict_from_smiles(self, smiles: List[str], parameters: dict):
        """
        :return: activity predictions and a set of valid indices indicating which are the valid smiles
        :param smiles: list of smiles strings
        :type parameters: this is a dictionary object that contains the boundary constants for
        scaling continuous values via sigmoid function. The expected keys are: "low" "high" and "k"
        to calculate the sigmoid scaling key "sigmoid" should be set to True
        """
        molecules, valid_idx = self._smiles_to_mols(smiles)
        fps = self._molecules_to_descriptors(molecules, parameters)

        if len(valid_idx) == 0:
            return np.empty([]), valid_idx
        activity = self.predict_from_fingerprints(fps, parameters)
        return activity, valid_idx

    def predict_from_mols(self, molecules: List, parameters: dict):
        """
        :return: activity predictions and a set of valid indices indicating which are the valid smiles
        :param molecules: list of RDKit molecule objects
        :type parameters: this is a dictionary object that contains the boundary constants for
        scaling continuous values via sigmoid function. The expected keys are: "low" "high" and "k"
        to calculate the sigmoid scaling key "sigmoid" should be set to True
        """

        if len(molecules) == 0:
            return np.empty([])
        fps = self._molecules_to_descriptors(molecules, parameters)
        activity = self.predict_from_fingerprints(fps, parameters)
        return activity

    def predict_from_fingerprints(self, fps, parameters):
        if self.__model_type == "regression":
            predicted_activity = self.__activity_model.predict(fps)
            activity = self._prediction_transformation(predicted_activity, parameters)
        else:
            predictions = self.__activity_model.predict_proba(fps)
            activity = predictions[:, 1]

        return activity

    def _prediction_transformation(self, predicted_activity, parameters: {}):
        if parameters.get("transformation", False):
            activity = self.transformation(predicted_activity, parameters)
        else:
            activity = predicted_activity
        return activity

    def _assign_transformation(self, specific_parameters: {}):
        """classification models should not have any prediction transformations"""
        if specific_parameters["scikit"] == "classification":
            specific_parameters["transformation"] = False
            specific_parameters["transformation_type"] = "no_transformation"
        factory = TransformationFactory()
        transform_function = factory.get_transformation_function(specific_parameters)
        return transform_function

    def _maccs_keys(self, molecules: List, parameters: {}):
        fingerprints = []
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in molecules]
        for fp in fps:
            fp_np = np.zeros((1,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints

    def _avalon(self, molecules: List, parameters: {}):
        size = parameters.get('size', 512)
        fingerprints = []
        fps = [pyAvalonTools.GetAvalonFP(mol) for mol in molecules]
        for fp in fps:
            fp_np = np.zeros((1, size), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints

    def _descriptor_registry(self) -> dict:
        descriptor_list = dict(ecfp=self._smiles_to_fingerprints,
                               ecfp_counts=self._smiles_to_count_fingerprints,
                               maccs_keys=self._maccs_keys,
                               avalon=self._avalon)
        return descriptor_list

    def _load_descriptor(self, parameters: {}):
        descriptor_type = parameters.get('descriptor_type', 'ecfp_counts')
        registry = self._descriptor_registry()
        descriptor = registry[descriptor_type]
        return descriptor
