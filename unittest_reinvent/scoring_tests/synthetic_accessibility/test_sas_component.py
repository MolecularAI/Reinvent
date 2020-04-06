import unittest
import os
import gzip
import shutil

import utils.smiles as chem_smiles
from scoring.score_components.synthetic_accessibility.sas_component import SASComponent
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, SAS_MODEL_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum


class Test_sas_component(unittest.TestCase):

    def setUp(self):
        csp_enum = ComponentSpecificParametersEnum()
        ts_parameters = create_activity_component_regression()
        ts_parameters.specific_parameters[csp_enum.TRANSFORMATION] = False

        # unzip the model for loading
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)
        tmp_model_path = os.path.join(MAIN_TEST_PATH, os.path.splitext(os.path.basename(SAS_MODEL_PATH))[0])
        with gzip.open(SAS_MODEL_PATH, "rb") as f_in:
            with open(tmp_model_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        ts_parameters.model_path = tmp_model_path

        self.query_smiles = ['n1cccc2ccccc12']
        self.query_mols = [chem_smiles.to_mol(smile) for smile in self.query_smiles]
        self.component = SASComponent(ts_parameters)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_sas_1(self):
        summary = self.component.calculate_score(self.query_mols)
        self.assertAlmostEqual(summary.total_score[0], 0.99, 3)
