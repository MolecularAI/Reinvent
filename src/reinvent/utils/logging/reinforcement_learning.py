import time
from operator import itemgetter

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def find_matching_pattern_in_smiles(list_of_mols: [], smarts_pattern=None) -> []:

    def orient_molecule_according_to_matching_pattern(molecule, pattern):
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None:
                AllChem.Compute2DCoords(pattern_mol)
                AllChem.GenerateDepictionMatching2DStructure(molecule, pattern_mol, acceptFailure=True)
        except:
            pass

    matches = []
    if smarts_pattern is not None:
        for mol in list_of_mols:
            if mol is not None:
                match_pattern = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts_pattern))
                orient_molecule_according_to_matching_pattern(mol, smarts_pattern) if len(match_pattern) > 0 else ()
                matches.append(match_pattern)
            else:
                no_pattern = ()
                matches.append(no_pattern)
    return matches


def padding_with_invalid_smiles(smiles, sample_size):
    diff = len(smiles) - sample_size
    if diff < 0:
        bulk = ["INVALID" for _ in range(-diff)]
        bulk_np = np.array(bulk)
        smiles = np.concatenate((smiles, bulk_np))
    return smiles


def check_for_invalid_mols_and_create_legend(smiles, score, sample_size):
    legends = []
    list_of_mols = []
    for i in range(sample_size):
        list_of_mols.extend([Chem.MolFromSmiles(smiles[i])])
        if list_of_mols[i] is not None:
            legends.extend([f"Score:{score[i].item():.3f}"])
        elif list_of_mols[i] is None:
            legends.extend([f"This Molecule Is Invalid"])
    return list_of_mols, legends


def sort_smiles_by_score(score, smiles: []):
    paired = []
    for indx, _ in enumerate(score):
        paired.append((score[indx], smiles[indx]))
    result = sorted(paired, key=itemgetter(0), reverse=True)
    sorted_score = []
    sorted_smiles = []
    for r in result:
        sorted_score.append(r[0])
        sorted_smiles.append(r[1])
    return sorted_score, sorted_smiles


def estimate_run_time(start_time, n_steps, step):
    time_elapsed = int(time.time() - start_time)
    time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
    summary = {"elapsed": time_elapsed, "left": time_left}
    return summary
