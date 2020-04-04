import unittest

from scoring.scoring_function_factory import ScoringFunctionFactory
from scoring.scoring_function_parameters import ScoringFuncionParameters
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.enums.scoring_function_enum import ScoringFunctionNameEnum


class Test_scoring_function_factory(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        enum = ScoringFunctionComponentNameEnum()
        ts_parameters = dict(component_type=enum.TANIMOTO_SIMILARITY,
                             name="tanimoto_similarity",
                             weight=1.,
                             smiles=["CCC", "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"],
                             model_path="",
                             specific_parameters={})
        sf_enum = ScoringFunctionNameEnum()
        sf_parameters = ScoringFuncionParameters(name=sf_enum.CUSTOM_SUM, parameters=[ts_parameters])
        self.sf_instance = ScoringFunctionFactory(sf_parameters=sf_parameters)

    def test_sf_factory_1(self):
        result = self.sf_instance.get_final_score(["CCC"])
        self.assertEqual(1., result.total_score)

    def test_sf_factory_2(self):
        result = self.sf_instance.get_final_score(["CCCCCC"])
        self.assertAlmostEqual(result.total_score[0], 0.353, 3)

