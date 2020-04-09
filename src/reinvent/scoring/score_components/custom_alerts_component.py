from rdkit import Chem
from typing import List

from ..component_parameters import ComponentParameters
from ..score_components.base_score_component import BaseScoreComponent
from ..score_summary import ComponentSummary


class CustomAlerts(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._set_custom_alerts()

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._substructure_match(molecules, self.custom_alerts)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _substructure_match(self, query_mols, list_of_SMARTS):
        match = [any([mol.HasSubstructMatch(Chem.MolFromSmarts(subst)) for subst in list_of_SMARTS
                      if Chem.MolFromSmarts(subst)]) for mol in query_mols]
        reverse = [1 - m for m in match]
        return reverse

    def _set_custom_alerts(self):
        self.custom_alerts = ['']
        if len(self.parameters.smiles) > 0:
            self.custom_alerts = self.parameters.smiles
