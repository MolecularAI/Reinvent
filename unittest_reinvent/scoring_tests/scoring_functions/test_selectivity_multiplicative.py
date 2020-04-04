import unittest

from scoring.component_parameters import ComponentParameters
from scoring.function import CustomProduct
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression, \
    create_custom_alerts_configuration
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class Test_selectivity_multiplicative_function(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        enum = ScoringFunctionComponentNameEnum()
        activity = create_activity_component_regression()
        qed_score = ComponentParameters(component_type=enum.QED_SCORE,
                                        name="qed_score_name",
                                        weight=1.,
                                        smiles=[],
                                        model_path="",
                                        specific_parameters={})

        custom_alerts = create_custom_alerts_configuration()

        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure_name",
                                                    weight=1.,
                                                    smiles=["*CCC"],
                                                    model_path="",
                                                    specific_parameters={})
        self.sf_state = CustomProduct(
            parameters=[activity, qed_score, custom_alerts, matching_substructure])

    def test_special_selectivity_multiplicative_1(self):
        score = self.sf_state.get_final_score(smiles=["O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"])
        self.assertAlmostEqual(score.total_score[0], 0.010293521, 3)

