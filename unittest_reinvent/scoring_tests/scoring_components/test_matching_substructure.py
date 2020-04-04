import unittest

from scoring.component_parameters import ComponentParameters
from scoring.score_components import MatchingSubstructure
from unittest_reinvent.scoring_tests.scoring_components import ScoringTest
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
import numpy.testing as npt


class Test_matching_substructures(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sf_enum = ScoringFunctionComponentNameEnum()
        parameters = ComponentParameters(component_type=sf_enum.MATCHING_SUBSTRUCTURE,
                                         name="matching_substructure",
                                         weight=1.,
                                         smiles=["c1ccccc1"],
                                         model_path="",
                                         specific_parameters={})
        cls.component = MatchingSubstructure(parameters)

    def test_match_1(self):
        npt.assert_almost_equal(self.score("Cn1cc(c([NH])cc1=O)"), 0.5)

    def test_match_2(self):
        npt.assert_almost_equal(self.score("CCC"), 0.5)

    def test_match_3(self):
        npt.assert_almost_equal(self.score("c1ccccc1CC"), 1.0)


class Test_matching_substructures_not_provided(ScoringTest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sf_enum = ScoringFunctionComponentNameEnum()
        parameters = ComponentParameters(component_type=sf_enum.MATCHING_SUBSTRUCTURE,
                                         name="matching_substructure",
                                         weight=1.,
                                         smiles=[],
                                         model_path="",
                                         specific_parameters={})
        cls.component = MatchingSubstructure(parameters)

    def test_match_no_structure_1(self):
        npt.assert_almost_equal(self.score("c1ccccc1CC"), 1.0)


class Test_invalid_matching_substructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sf_enum = ScoringFunctionComponentNameEnum()
        cls.matching_pattern = "XXXX"
        cls.parameters = ComponentParameters(component_type=sf_enum.MATCHING_SUBSTRUCTURE,
                                             name="matching_substructure",
                                             weight=1.,
                                             smiles=[cls.matching_pattern],
                                             model_path="",
                                             specific_parameters={})

    def test_match_invalid_structure_1(self):
        with self.assertRaises(IOError) as context:
            _ = MatchingSubstructure(self.parameters)
        msg = f"Invalid smarts pattern provided as a matching substructure: {self.matching_pattern}"
        self.assertEqual(msg, str(context.exception))
