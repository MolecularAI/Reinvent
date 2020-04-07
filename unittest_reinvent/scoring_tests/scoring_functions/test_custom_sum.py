import unittest

from scoring.component_parameters import ComponentParameters
from scoring.function import CustomSum
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import \
    create_predictive_property_component_regression, create_activity_component_regression
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class Test_custom_sum(unittest.TestCase):

    def setUp(self):
        enum = ScoringFunctionComponentNameEnum()
        predictive_property = create_predictive_property_component_regression()
        activity = create_activity_component_regression()
        qed_score = ComponentParameters(component_type=enum.QED_SCORE,
                                        name="qed_score_name",
                                        weight=1.,
                                        smiles=[],
                                        model_path="",
                                        specific_parameters={})
        custom_alerts = ComponentParameters(component_type=enum.CUSTOM_ALERTS,
                                            name="custom_alerts_name",
                                            weight=1.,
                                            smiles=["CCCOOO"],
                                            model_path="",
                                            specific_parameters={})
        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure_name",
                                                    weight=1.,
                                                    smiles=["*CCC"],
                                                    model_path="",
                                                    specific_parameters={})
        self.sf_instance = CustomSum(
            parameters=[activity, qed_score, custom_alerts, matching_substructure, predictive_property])

    def test_special_selectivity_multiplicative_no_sigm_trans_1(self):
        score = self.sf_instance.get_final_score(smiles=["O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"])
        self.assertAlmostEqual(score.total_score[0], 0.3631, 3)

    def test_special_selectivity_multiplicative_no_sigm_trans_3(self):
        score = self.sf_instance.get_final_score(smiles=["CCCOOOCCCOOO"])
        self.assertEqual(score.total_score, 0)



