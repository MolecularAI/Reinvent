from rdkit.Chem.Descriptors import MolWt
from ...component_parameters import ComponentParameters
from .base_physchem_component import BasePhysChemComponent


class MolWeight(BasePhysChemComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def _calculate_phys_chem_property(self, mol):
        return MolWt(mol)
