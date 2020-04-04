import unittest

import utils.smiles as chem_smiles
from scoring.score_components.synthetic_accessibility.sas_component import SASComponent
from unittest_reinvent.fixtures.paths import SAS_MODEL_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum


# FIXME: remember to unpack the SA_score_prediction.7z for this test to work
class Test_sas_component(unittest.TestCase):

    def setUp(self):
        csp_enum = ComponentSpecificParametersEnum()
        ts_parameters = create_activity_component_regression()
        ts_parameters.specific_parameters[csp_enum.TRANSFORMATION] = False
        ts_parameters.model_path = SAS_MODEL_PATH

        self.query_smiles = ['n1cccc2ccccc12']
        self.query_mols = [chem_smiles.to_mol(smile) for smile in self.query_smiles]
        self.component = SASComponent(ts_parameters)

    def test_sas_1(self):
        summary = self.component.calculate_score(self.query_mols)
        self.assertAlmostEqual(summary.total_score[0], 0.99, 3)

