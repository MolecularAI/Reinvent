import unittest

from scoring.score_components import PredictivePropertyComponent
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import \
    create_predictive_property_component_regression
from unittest_reinvent.scoring_tests.scoring_components import ScoringTest
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
import numpy.testing as npt


class Test_predictive_property_component(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.csp_enum = ComponentSpecificParametersEnum()
        activity = create_predictive_property_component_regression()
        cls.component = PredictivePropertyComponent(activity)

    def test_predictive_property_1(self):
        npt.assert_almost_equal(self.score("O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"), 0.7125064, 3)

    def test_predictive_property_2(self):
        self.assertTrue(self.component.parameters.specific_parameters[self.csp_enum.TRANSFORMATION])