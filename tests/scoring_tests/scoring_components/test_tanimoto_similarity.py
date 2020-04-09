import unittest

from scoring.component_parameters import ComponentParameters
from scoring.score_components import TanimotoSimilarity
from unittest_reinvent.scoring_tests.scoring_components import ScoringTest
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
import numpy.testing as npt


class Test_tanimoto_similarity(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        enum = ScoringFunctionComponentNameEnum()
        parameters = ComponentParameters(component_type=enum.TANIMOTO_SIMILARITY,
                                         name="tanimoto_similarity",
                                         weight=1.,
                                         smiles=["CCC", "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"],
                                         model_path="",
                                         specific_parameters={})
        cls.component = TanimotoSimilarity(parameters)

    def test_similarity_1(self):
        npt.assert_almost_equal(self.score("CCC"), 1.0)

    def test_similarity_2(self):
        npt.assert_almost_equal(self.score("O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"), 1.0)

    def test_similarity_3(self):
        npt.assert_array_less(self.score("Cn1cc(c([NH])cc1=O)"), 0.5)

    def test_similarity_4(self):
        smiles = ["CCC", "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"]
        scores = self.multiple_scores(smiles)
        npt.assert_almost_equal(scores, 1.0,)
