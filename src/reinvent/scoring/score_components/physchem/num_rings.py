from rdkit.Chem.rdMolDescriptors import CalcNumRings
from ...component_parameters import ComponentParameters
from .base_physchem_component import BasePhysChemComponent


class NumRings(BasePhysChemComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def _calculate_phys_chem_property(self, mol):
        return CalcNumRings(mol)
