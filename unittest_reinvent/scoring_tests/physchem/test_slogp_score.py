import unittest

import numpy as np
import numpy.testing as npt

from scoring.component_parameters import ComponentParameters
from scoring.function import CustomSum
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.enums.transformation_type_enum import TransformationTypeEnum


class Test_slogp_score_no_transformation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        ts_parameters = ComponentParameters(component_type=sf_enum.SLOGP,
                                            name="SlogP",
                                            weight=1.,
                                            smiles=[],
                                            model_path="",
                                            specific_parameters={
                                                csp_enum.TRANSFORMATION: False
                                            })
        self.sf_state = CustomSum(parameters=[ts_parameters])


    def test_slogp_1(self):
        smiles = [
                  "OC(=O)P(=O)(O)O",
                  "Cc1ccccc1N1C(=O)c2cc(S(N)(=O)=O)c(Cl)cc2NC1C",
                  "N12CC3C(=NC4C(C=3)=CC=CC=4)C1=CC1=C(COC(=O)C1(O)CC)C2=O",
                  "N12CC3C(=NC4C(C=3C(=O)O)=CC3=C(OCCO3)C=4)C1=CC1=C(COC(=O)C1(O)CC)C2=O",
                  "FC1C=CC(CC(=NS(=O)(=O)C2C=CC(C)=CC=2)N2CCN(CC3C4C(=CC=CC=4)N=C4C=3CN3C4=CC4=C(COC(=O)C4(O)CC)C3=O)CC2)=CC=1"
                  ]
        values = np.array([-0.1579, 2.71412, 2.0796, 1.549, 4.67482])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)

class Test_slogp_score_with_double_sigmoid(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        tt_enum = TransformationTypeEnum()
        specific_parameters = {
                                csp_enum.TRANSFORMATION: True,
                                csp_enum.LOW: 1,
                                csp_enum.HIGH: 3,
                                csp_enum.COEF_DIV: 3,
                                csp_enum.COEF_SI: 10,
                                csp_enum.COEF_SE: 10,
                                csp_enum.TRANSFORMATION_TYPE: tt_enum.DOUBLE_SIGMOID
                               }
        ts_parameters = ComponentParameters(component_type=sf_enum.SLOGP,
                                            name="SlogP",
                                            weight=1.,
                                            smiles=[],
                                            model_path="",
                                            specific_parameters=specific_parameters
                                            )
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_slogp_1(self):
        smiles = [
                  "OC(=O)P(=O)(O)O",
                  "Cc1ccccc1N1C(=O)c2cc(S(N)(=O)=O)c(Cl)cc2NC1C",
                  "N12CC3C(=NC4C(C=3)=CC=CC=4)C1=CC1=C(COC(=O)C1(O)CC)C2=O",
                  "N12CC3C(=NC4C(C=3C(=O)O)=CC3=C(OCCO3)C=4)C1=CC1=C(COC(=O)C1(O)CC)C2=O",
                  "FC1C=CC(CC(=NS(=O)(=O)C2C=CC(C)=CC=2)N2CCN(CC3C4C(=CC=CC=4)N=C4C=3CN3C4=CC4=C(COC(=O)C4(O)CC)C3=O)CC2)=CC=1"
                  ]
        values = np.array([0.0, 0.9, 1.0, 1.0, 0.0])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
