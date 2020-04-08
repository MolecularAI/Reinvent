from typing import List

from scoring.component_parameters import ComponentParameters
from scoring.score_components import TanimotoSimilarity, \
    JaccardDistance, CustomAlerts, QedScore, MatchingSubstructure, \
    PredictivePropertyComponent, SelectivityComponent, \
    SASComponent, MolWeight, PSA, RotatableBonds, HBD_Lipinski, NumRings
from scoring.score_components.base_score_component import BaseScoreComponent
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class ScoreComponentFactory:
    def __init__(self, parameters: List[ComponentParameters]):
        self._parameters = parameters
        self._current_components = self._default_scoring_component_registry()

    def _default_scoring_component_registry(self) -> dict:
        enum = ScoringFunctionComponentNameEnum()
        component_map = {
            enum.MATCHING_SUBSTRUCTURE: MatchingSubstructure,
            enum.PREDICTIVE_PROPERTY: PredictivePropertyComponent,
            enum.TANIMOTO_SIMILARITY: TanimotoSimilarity,
            enum.JACCARD_DISTANCE: JaccardDistance,
            enum.CUSTOM_ALERTS: CustomAlerts,
            enum.QED_SCORE: QedScore,
            enum.MOLECULAR_WEIGHT: MolWeight,
            enum.TPSA: PSA,
            enum.NUM_ROTATABLE_BONDS: RotatableBonds,
            enum.NUM_HBD_LIPINSKI: HBD_Lipinski,
            enum.NUM_RINGS: NumRings,
            enum.SELECTIVITY: SelectivityComponent,
            enum.SA_SCORE: SASComponent
        }
        return component_map

    def create_score_components(self) -> [BaseScoreComponent]:
        return [self._current_components.get(p.component_type)(p) for p in self._parameters]
