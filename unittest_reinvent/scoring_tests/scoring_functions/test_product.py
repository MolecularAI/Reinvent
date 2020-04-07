import unittest

from scoring.component_parameters import ComponentParameters
from scoring.function import CustomProduct
from unittest_reinvent.scoring_tests.fixtures import create_activity_component_regression, \
    create_predictive_property_component_regression
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class Test_product(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        predictive_property = create_predictive_property_component_regression()
        activity = create_activity_component_regression()
        qed_score = ComponentParameters(component_type=sf_enum.QED_SCORE,
                                        name="qed_score_name",
                                        weight=1.,
                                        smiles=[],
                                        model_path="",
                                        specific_parameters={})
        self.sf_state = CustomProduct(parameters=[activity, qed_score, predictive_property])

    def test_product_1(self):
        score = self.sf_state.get_final_score(smiles=["O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"])
        self.assertAlmostEqual(score.total_score[0], 0.726, 3)
