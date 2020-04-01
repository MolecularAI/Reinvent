import numpy as np
import torch
from rdkit import Chem


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def get_indices_of_unique_smiles(smiles: [str]) -> np.array:
    """Returns an np.array of indices corresponding to the first entries in a list of smiles strings"""
    _, idxs = np.unique(smiles, return_index=True)
    sorted_indices = np.sort(idxs)
    return sorted_indices


def set_default_device_cuda():
    """Sets the default device (cpu or cuda) used for all tensors."""
    if torch.cuda.is_available() == False:
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return False
    else:  # device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return True


def fraction_valid_smiles(smiles):
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    fraction = 100 * i / len(smiles)
    return fraction
