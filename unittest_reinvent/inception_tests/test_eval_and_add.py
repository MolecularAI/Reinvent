import os
import shutil
import unittest

import numpy as np
import numpy.testing as nt
import torch as ts

import utils.general as utils_general
from models.model import Model
from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.reinforcement_learning.inception import Inception
from scoring.component_parameters import ComponentParameters
from scoring.function import CustomSum
from unittest_reinvent.fixtures.paths import RANDOM_PRIOR_PATH, MAIN_TEST_PATH
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


# TODO: this really needs to be refactored
class Test_inception_eval_add(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        utils_general.set_default_device_cuda()
        self.log_path = MAIN_TEST_PATH

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        smiles = np.array(['CCC', 'CCCC', 'CCCCCC'])
        score = [0, 0.5, 1.]
        prior_likelihood = ts.tensor([0, 10, 100])

        prior = Model.load_from_file(RANDOM_PRIOR_PATH)
        config = InceptionConfiguration(smiles=smiles, memory_size=4, sample_size=4)
        scoring = ComponentParameters(component_type=sf_enum.JACCARD_DISTANCE,
                                      name="jaccard_distance",
                                      weight=1.,
                                      smiles=["CONN", "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"],
                                      model_path="",
                                      specific_parameters={})
        scoring_function = CustomSum(parameters=[scoring])

        self.inception_model = Inception(configuration=config, scoring_function=scoring_function, prior=prior)

        self.inception_model.add(smiles, score, prior_likelihood)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.log_path):
            shutil.rmtree(self.log_path)

    def test_eval_add_1(self):
        self.assertEqual(len(self.inception_model.memory), 3)

    def test_eval_add_2(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        smiles = np.array(['CCC', 'CCCC', 'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'])
        scoring = ComponentParameters(component_type=sf_enum.TANIMOTO_SIMILARITY,
                                      name="tanimoto_similarity",
                                      weight=1.,
                                      smiles=["CCC", "CC"],
                                      model_path="",
                                      specific_parameters={})
        scoringfunction = CustomSum(parameters=[scoring])
        prior = Model.load_from_file(RANDOM_PRIOR_PATH)
        self.inception_model.evaluate_and_add(smiles, scoringfunction, prior)
        self.assertEqual(len(self.inception_model.memory), 4)
        nt.assert_almost_equal(np.array(self.inception_model.memory['score'].values),
                               np.array([0.96, 0.9412, 0.9286, 0.0345]), 4)
