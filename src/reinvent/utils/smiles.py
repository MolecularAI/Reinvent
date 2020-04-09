# coding=utf-8

import random

import rdkit.Chem as rkc
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import SaltRemover
from rdkit.Chem import rdmolops


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()


def _initialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(rkc.MolFromSmarts(x), rkc.MolFromSmiles(y, False)) for x, y in patts]


def _neutralise_charges(mol, reactions=None):
    if reactions is None:
        reactions = _initialiseNeutralisationReactions()
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return mol, True
    else:
        return mol, False


def _get_largest_fragment(mol):
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    maxmol = None
    for mol in frags:
        if mol is None:
            continue
        if maxmol is None:
            maxmol = mol
        if maxmol.GetNumHeavyAtoms() < mol.GetNumHeavyAtoms():
            maxmol = mol
    return maxmol


def _remove_salts(mol):
    return SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)


def _valid_size(mol, min_heavy_atoms, max_heavy_atoms, element_list, remove_long_side_chains):
    """Filters molecules on number of heavy atoms and atom types"""
    mol = _rare_filters(mol)
    if mol:
        correct_size = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        if not correct_size:
            return

        valid_elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if not valid_elements:
            return

        has_long_sidechains = False
        if remove_long_side_chains:
            # remove aliphatic side chains with at least 5 carbons not in a ring
            sma = '[CR0]-[CR0]-[CR0]-[CR0]-[CR0]'
            has_long_sidechains = mol.HasSubstructMatch(rkc.MolFromSmarts(sma))

        return correct_size and valid_elements and not has_long_sidechains


def _rare_filters(mol):
    if mol:
        cyano_filter = "[C-]#[N+]"
        oh_filter = "[OH+]"
        sulfur_filter = "[SH]"
        if not mol.HasSubstructMatch(rkc.MolFromSmarts(cyano_filter)) \
                and not mol.HasSubstructMatch(rkc.MolFromSmarts(oh_filter)) \
                and not mol.HasSubstructMatch(rkc.MolFromSmarts(sulfur_filter)):
            return mol


def standardize_smiles(smiles, min_heavy_atoms=2, max_heavy_atoms=70, element_list=None,
                       remove_long_side_chains=True, neutralise_charges=True):
    if element_list is None:
        element_list = [6, 7, 8, 9, 16, 17, 35]
    mol = rkc.MolFromSmiles(smiles)
    if mol:
        mol = _get_largest_fragment(mol)
    if mol:
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol:
        mol = _remove_salts(mol)
    if mol and neutralise_charges:
        mol, _ = _neutralise_charges(mol)
    if mol:
        rdmolops.Cleanup(mol)
        rdmolops.SanitizeMol(mol)
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol and _valid_size(mol, min_heavy_atoms, max_heavy_atoms, element_list, remove_long_side_chains):
        return rkc.MolToSmiles(mol, isomericSmiles=False)
    return None


def randomize_smiles(smiles, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param smiles: A smiles string
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    mol = rkc.MolFromSmiles(smiles)
    if not mol:
        return None

    if random_type == "unrestricted":
        return rkc.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumHeavyAtoms()))
        random.shuffle(new_atom_order)
        random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
        return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError("Type '{}' is not valid".format(random_type))


def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smi:
        return rkc.MolFromSmiles(smi)


def convert_to_rdkit_smiles(smiles, allowTautomers=True, sanitize=False, isomericSmiles=False):
    """
    Converts a Mol object into a canonical SMILES string.
    :type allowTautomers: allows having same molecule represented in different tautomeric forms
    """
    if allowTautomers:
        return rkc.MolToSmiles(rkc.MolFromSmiles(smiles, sanitize=sanitize), isomericSmiles=isomericSmiles)
    else:
        return MolStandardize.canonicalize_tautomer_smiles(smiles)


def read_smiles_file(file_path, ignore_invalid=True, num=-1, standardize=True, randomize=False):
    """
    Reads a SMILES file.
    :param randomize: Standardizes smiles.
    :param standardize: Randomizes smiles.
    :param file_path: Path to a SMILES file.
    :param ignore_invalid: Ignores invalid lines (empty lines)
    :param num: Parse up to num rows.
    :return: An iterator with the rows.
    """
    actions = []
    if standardize:
        actions.append(standardize_smiles)
    if randomize:
        actions.append(randomize_smiles)

    with open(file_path, "r") as csv_file:
        for i, row in enumerate(csv_file):
            if i == num:
                break
            line = row.rstrip().replace(",", " ").split()
            smiles = line[0]
            for action in actions:
                if smiles:
                    smiles = action(smiles)
            if smiles:
                yield smiles
            elif not ignore_invalid:
                yield None
