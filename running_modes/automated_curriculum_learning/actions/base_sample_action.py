import numpy as np
from running_modes.automated_curriculum_learning.actions import BaseAction


class BaseSampleAction(BaseAction):

    def _get_indices_of_unique_smiles(self, smiles: [str]) -> np.array:
        """Returns an np.array of indices corresponding to the first entries in a list of smiles strings"""
        _, idxs = np.unique(smiles, return_index=True)
        sorted_indices = np.sort(idxs)
        return sorted_indices