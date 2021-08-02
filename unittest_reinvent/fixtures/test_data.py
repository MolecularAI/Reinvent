import numpy as np

INVALID = 'INVALID'
NONSENSE = 'C1CC(Br)CCC1[ClH]'
ASPIRIN='O=C(C)Oc1ccccc1C(=O)O'
CELECOXIB='O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
IBUPROFEN='CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
ETHANE = 'CC'
PROPANE='CCC'
BUTANE='CCCC'
PENTANE = 'CCCCC'
HEXANE = 'CCCCCC'
METAMIZOLE='CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CS(=O)(=O)O'
CAFFEINE='CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
COCAINE='CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC'
BENZENE='c1ccccc1'
TOLUENE='c1ccccc1C'
ANILINE='c1ccccc1N'
AMOXAPINE = 'C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl'
GENTAMICIN = 'CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC'
METHOXYHYDRAZINE = 'CONN'
HYDROPEROXYMETHANE = 'COO'
SCAFFOLD_SUZUKI = 'Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*]'
CELECOXIB_FRAGMENT = "*c1cc(C(F)(F)F)nn1-c1ccc(S(N)(=O)=O)cc1"
REACTION_SUZUKI = "[*;$(c2aaaaa2),$(c2aaaa2):1]-!@[*;$(c2aaaaa2),$(c2aaaa2):2]>>[*:1][*].[*:2][*]"
DECORATION_SUZUKI = '[*]c1ncncc1'
TWO_DECORATIONS_SUZUKI = '[*]c1ncncc1|[*]c1ncncc1'
TWO_DECORATIONS_ONE_SUZUKI = '[*]c1ncncc1|[*]C'
SCAFFOLD_NO_SUZUKI = '[*:0]Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)'
DECORATION_NO_SUZUKI = '[*]C'
CELECOXIB_SCAFFOLD = 'Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*:0]'
SCAFFOLD_TO_DECORATE = "[*]c1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*]"


REP_LIKELIHOOD = np.array(
    [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 10.0, 10.0, 10.0,
     10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

LIKELIHOODLIST = np.array(
    [20.0, 20.0, 19.0, 19.0, 18.0, 18.0, 20.0, 21.0, 21.0, 20.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 20.0, 21.0,
     22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 20.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 20.0, 21.0, 22.0, 23.0, 24.0,
     25.0, 26.0, 27.0, 20.0], dtype=np.float32)

INVALID_SMILES_LIST = [INVALID] * 25
REP_SMILES_LIST = [ASPIRIN] * 15 + [CELECOXIB] * 10