import unittest

from rdkit import Chem

from scoring.component_parameters import ComponentParameters
from scoring.score_components import QedScore
from unittest_reinvent.scoring_tests.scoring_components import ScoringTest
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
import numpy.testing as npt


class Test_qed_score(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sf_enum = ScoringFunctionComponentNameEnum()
        parameters = ComponentParameters(component_type=sf_enum.QED_SCORE,
                                         name="qed_score",
                                         weight=1.,
                                         smiles=[],
                                         model_path="",
                                         specific_parameters={})
        cls.component = QedScore(parameters)
        cls.smile = "C1CCCCCCCCC1"
        cls.mol = Chem.MolFromSmiles(cls.smile)

    def test_molecule_parsed_successfully(self):
        self.assertIsNotNone(self.mol)

    def test_invalid_molecule_returns_zero(self):
        score = self.component.calculate_score([None])
        npt.assert_almost_equal(score.total_score[0], 0.0, 4)

    def test_one_molecule(self):
        score = self.component.calculate_score([self.mol])
        self.assertEqual(1, len(score.total_score))
        npt.assert_almost_equal(score.total_score[0], 0.4784, 4)

    def test_one_molecule_2(self):
        npt.assert_almost_equal(self.score(self.smile), 0.4784, 3)

    def test_two_molecules(self):
        score = self.component.calculate_score([self.mol, self.mol])
        self.assertEqual(2, len(score.total_score))
        npt.assert_almost_equal(score.total_score[0], 0.4784, 4)
        npt.assert_almost_equal(score.total_score[1], 0.4784, 4)
