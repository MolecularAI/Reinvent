import unittest

from scoring.component_parameters import ComponentParameters
from scoring.score_components import SelectivityComponent
from unittest_reinvent.fixtures.paths import SCIKIT_REGRESSION_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression, \
    create_offtarget_activity_component_regression, create_offtarget_activity_component_classification, \
    create_activity_component_classification
from unittest_reinvent.scoring_tests.scoring_components import ScoringTest
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.enums.transformation_type_enum import TransformationTypeEnum
import numpy.testing as npt


class Test_mixed_selectivity_component(ScoringTest, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        enum = ScoringFunctionComponentNameEnum()

        delta_params = {
            "high": 3.0,
            "k": 0.25,
            "low": 0.0,
            "transformation": True,
            "transformation_type": "sigmoid"
        }

        activity = create_activity_component_regression()
        activity.specific_parameters[csp_enum.TRANSFORMATION_TYPE] = transf_type.DOUBLE_SIGMOID
        activity.specific_parameters[csp_enum.COEF_DIV] = 100.
        activity.specific_parameters[csp_enum.COEF_SI] = 150.
        activity.specific_parameters[csp_enum.COEF_SE] = 150.

        off_activity = create_offtarget_activity_component_classification()

        selectivity = ComponentParameters(component_type=enum.SELECTIVITY,
                                          name="desirability",
                                          weight=1.,
                                          smiles=[],
                                          model_path="",
                                          specific_parameters={
                                               "activity_model_path": activity.model_path,
                                               "offtarget_model_path": off_activity.model_path,
                                               "activity_specific_parameters": activity.specific_parameters.copy(),
                                               "offtarget_specific_parameters": off_activity.specific_parameters.copy(),
                                               "delta_transformation_parameters": delta_params
                                          })
        self.component = SelectivityComponent(parameters=selectivity)

    def test_selectivity_component(self):
        smiles = ["O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N", "CCCC"]
        expected_values = [0.01, 0.01]
        scores = self.multiple_scores(smiles)
        npt.assert_almost_equal(scores, expected_values, decimal=3)


class Test_classification_selectivity_component(ScoringTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        enum = ScoringFunctionComponentNameEnum()

        delta_params = {
            "high": 3.0,
            "k": 0.25,
            "low": 0.0,
            "transformation": True,
            "transformation_type": "sigmoid"
        }
        activity = create_activity_component_classification()
        activity.specific_parameters[csp_enum.TRANSFORMATION_TYPE] = transf_type.DOUBLE_SIGMOID
        activity.specific_parameters[csp_enum.COEF_DIV] = 100.
        activity.specific_parameters[csp_enum.COEF_SI] = 150.
        activity.specific_parameters[csp_enum.COEF_SE] = 150.

        off_activity = create_offtarget_activity_component_classification()

        selectivity = ComponentParameters(component_type=enum.SELECTIVITY,
                                          name="desirability",
                                          weight=1.,
                                          smiles=[],
                                          model_path="",
                                          specific_parameters={
                                               "activity_model_path": activity.model_path,
                                               "offtarget_model_path": off_activity.model_path,
                                               "activity_specific_parameters": activity.specific_parameters.copy(),
                                               "offtarget_specific_parameters": off_activity.specific_parameters.copy(),
                                               "delta_transformation_parameters": delta_params
                                           })

        cls.component = SelectivityComponent(parameters=selectivity)

    def test_selectivity_component_1(self):
        npt.assert_almost_equal(self.score("O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"), 0.01)


class Test_regression_selectivity_component(ScoringTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        enum = ScoringFunctionComponentNameEnum()

        delta_params = {
            "high": 3.0,
            "k": 0.25,
            "low": 0.0,
            "transformation": True,
            "transformation_type": "sigmoid"
        }

        activity = create_activity_component_regression()
        activity.specific_parameters[csp_enum.TRANSFORMATION_TYPE] = transf_type.DOUBLE_SIGMOID
        activity.specific_parameters[csp_enum.COEF_DIV] = 100.
        activity.specific_parameters[csp_enum.COEF_SI] = 150.
        activity.specific_parameters[csp_enum.COEF_SE] = 150.

        off_activity = create_offtarget_activity_component_regression()

        selectivity = ComponentParameters(component_type=enum.SELECTIVITY,
                                          name="desirability",
                                          weight=1.,
                                          smiles=[],
                                          model_path="",
                                          specific_parameters={
                                               "activity_model_path": activity.model_path,
                                               "offtarget_model_path": SCIKIT_REGRESSION_PATH,
                                               "activity_specific_parameters": activity.specific_parameters.copy(),
                                               "offtarget_specific_parameters": off_activity.specific_parameters.copy(),
                                               "delta_transformation_parameters": delta_params
                                           })
        cls.component = SelectivityComponent(parameters=selectivity)

    def test_selectivity_component(self):
        smiles = ["Cn1cc(c([NH])cc1=O)", "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"]
        expected_values = [0.01, 0.01]
        scores = self.multiple_scores(smiles)
        npt.assert_almost_equal(scores, expected_values, decimal=3)
