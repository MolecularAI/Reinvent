import unittest

from scoring.component_parameters import ComponentParameters
from scoring.score_components import CustomAlerts
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_custom_alerts_configuration
from unittest_reinvent.scoring_tests.scoring_components import ScoringTest
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
import numpy.testing as npt

class Test_custom_alerts_with_default_alerts(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parameters = create_custom_alerts_configuration()
        cls.component = CustomAlerts(parameters)

    def test_alert_1(self):
        npt.assert_almost_equal(self.multiple_scores(["C1CCCCCCCCC1", "C1CCCCCCCCCC1"]), [0.0, 0.0])

    def test_alert_2(self):
        npt.assert_almost_equal(self.score("O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"), 1.0)


class Test_custom_alerts_with_user_alerts(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        list_of_alerts = ["O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"]
        parameters = ComponentParameters(component_type=sf_enum.CUSTOM_ALERTS,
                                         name="custom_alerts",
                                         weight=1.,
                                         smiles=list_of_alerts,
                                         model_path="",
                                         specific_parameters={})
        self.component = CustomAlerts(parameters)

    def test_user_alert_1(self):
        npt.assert_almost_equal(self.multiple_scores(["C1CCCCCCCCC1", "C1CCCCCCCCCC1"]), [1.0, 1.0])

    def test_user_alert_2(self):
        npt.assert_almost_equal(self.score("O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"), 0.0)

