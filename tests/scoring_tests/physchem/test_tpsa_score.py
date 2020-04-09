import unittest

import numpy as np
import numpy.testing as npt

from scoring.component_parameters import ComponentParameters
from scoring.function import CustomSum
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.enums.transformation_type_enum import TransformationTypeEnum


class Test_tpsa_score_no_transformation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        ts_parameters = ComponentParameters(component_type=sf_enum.TPSA,
                                            name="TPSA",
                                            weight=1.,
                                            smiles=[],
                                            model_path="",
                                            specific_parameters={
                                                csp_enum.TRANSFORMATION: False
                                            })
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_tpsa_1(self):
        smiles = [
            "OC(=O)P(=O)(O)O",
            "C12C3C4C1C5C2C3C45",
            '[NH4+].[Cl-]',
            'n1cccc2ccccc12',
            'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
        ]
        values = np.array([94.83, 0., 36.5, 12.89, 77.98])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)


class Test_tpsa_score_with_double_sigmoid(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        tt_enum = TransformationTypeEnum()
        specific_parameters = {
            csp_enum.TRANSFORMATION: True,
            csp_enum.LOW: 0,
            csp_enum.HIGH: 135,
            csp_enum.COEF_DIV: 100,
            csp_enum.COEF_SI: 200,
            csp_enum.COEF_SE: 200,
            csp_enum.TRANSFORMATION_TYPE: tt_enum.DOUBLE_SIGMOID
        }
        ts_parameters = ComponentParameters(component_type=sf_enum.TPSA,
                                            name="TPSA",
                                            weight=1.,
                                            smiles=[],
                                            model_path="",
                                            specific_parameters=specific_parameters
                                            )
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_tpsa_1(self):
        smiles = [
            "OC(=O)P(=O)(O)O",
            "C12C3C4C1C5C2C3C45",
            '[NH4+].[Cl-]',
            'n1cccc2ccccc12',
            'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
        ]
        values = np.array([1., 0.5, 1., 1., 1.])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
