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
        ts_parameters = ComponentParameters(component_type=sf_enum.NUM_HBD_LIPINSKI,
                                            name="NumHBD_Lipinski",
                                            weight=1.,
                                            smiles=[],
                                            model_path="",
                                            specific_parameters={
                                                csp_enum.TRANSFORMATION: False
                                            })
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_hbd_1(self):
        smiles = [
                  "C(=O)N",
                  'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
                  ]
        values = np.array([1., 1.])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_equal(score.total_score, values)


class Test_tpsa_score_with_double_sigmoid(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        tt_enum = TransformationTypeEnum()
        specific_parameters = {
                                csp_enum.TRANSFORMATION: True,
                                csp_enum.LOW: 0,
                                csp_enum.HIGH: 1,
                                csp_enum.TRANSFORMATION_TYPE: tt_enum.STEP
                               }
        ts_parameters = ComponentParameters(component_type=sf_enum.NUM_HBD_LIPINSKI,
                                            name="NumHBD_Lipinski",
                                            weight=1.,
                                            smiles=[],
                                            model_path="",
                                            specific_parameters=specific_parameters
                                            )
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_hbd_1(self):
        smiles = [
                  "C(=O)N",
                  'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
                  ]
        values = np.array([1.0, 1.0])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_equal(score.total_score, values)



