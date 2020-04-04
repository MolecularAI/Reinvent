import unittest

from scoring.component_parameters import ComponentParameters
from scoring.function import CustomProduct
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression, \
    create_offtarget_activity_component_regression, create_custom_alerts_configuration
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.enums.transformation_type_enum import TransformationTypeEnum


class Test_desirability_multiplicative_function(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        sf_enum = ScoringFunctionComponentNameEnum()
        activity = create_activity_component_regression()
        activity.specific_parameters[csp_enum.TRANSFORMATION_TYPE] = transf_type.DOUBLE_SIGMOID
        activity.specific_parameters[csp_enum.COEF_DIV] = 100.
        activity.specific_parameters[csp_enum.COEF_SI] = 150.
        activity.specific_parameters[csp_enum.COEF_SE] = 150.
        off_activity = create_offtarget_activity_component_regression()

        delta_params = {
            "high": 3.0,
            "k": 0.25,
            "low": 0.0,
            "transformation": True,
            "transformation_type": "sigmoid"
        }

        selectivity = ComponentParameters(component_type=sf_enum.SELECTIVITY,
                                           name="desirability",
                                           weight=1.,
                                           smiles=[],
                                           model_path="",
                                           specific_parameters={
                                               "activity_model_path": activity.model_path,
                                               "offtarget_model_path": off_activity.model_path,
                                               "activity_specific_parameters": activity.specific_parameters.copy(),
                                               "offtarget_specific_parameters": off_activity.specific_parameters,
                                               "delta_transformation_parameters": delta_params
                                           })

        qed_score = ComponentParameters(component_type=sf_enum.QED_SCORE,
                                        name="qed_score",
                                        weight=1.,
                                        smiles=[],
                                        model_path="",
                                        specific_parameters={})
        matching_substructure = ComponentParameters(component_type=sf_enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure",
                                                    weight=1.,
                                                    smiles=["[*]n1cccc1CC"],
                                                    model_path="",
                                                    specific_parameters={})

        custom_alerts = create_custom_alerts_configuration()

        self.sf_state = CustomProduct(
            parameters=[activity, selectivity, qed_score, matching_substructure, custom_alerts])

    def test_desirability_multiplicative_1(self):
        score = self.sf_state.get_final_score(smiles=["CCC"])
        self.assertAlmostEqual(score.total_score[0], 0.001368927, 3)

    def test_desirability_multiplicative_4(self):
        score = self.sf_state.get_final_score(smiles=["C1CCCCCCCCC1", "12"])
        self.assertAlmostEqual(score.total_score[0], 0., 3)
        self.assertEqual(score.total_score[1], 0)
