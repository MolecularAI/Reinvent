import unittest

from scoring.component_parameters import ComponentParameters
from scoring.score_components import JaccardDistance
from unittest_reinvent.scoring_tests.scoring_components import ScoringTest
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
import numpy.testing as npt


class Test_jaccard_distance(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sf_enum = ScoringFunctionComponentNameEnum()
        parameters = ComponentParameters(component_type=sf_enum.JACCARD_DISTANCE,
                                         name="jaccard_distance",
                                         weight=1.,
                                         smiles=["O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N", "CCCCC"],
                                         model_path="",
                                         specific_parameters={})
        cls.component = JaccardDistance(parameters)

    def test_distance_1(self):
        npt.assert_almost_equal(self.score("CCCCC"), 0.0)

    def test_distance_2(self):
        npt.assert_almost_equal(self.score("O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"), 0.0)

    def test_distance_3(self):
        npt.assert_almost_equal(self.score("C1=CC2=C(C=C1)C1=CC=CC=C21"), 0.788, 3)


