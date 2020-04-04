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


class Test_empty_inception(unittest.TestCase):
    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        utils_general.set_default_device_cuda()
        self.log_path = MAIN_TEST_PATH

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        smiles = []
        score = []
        prior_likelihood = ts.tensor([])

        prior = Model.load_from_file(RANDOM_PRIOR_PATH)
        config = InceptionConfiguration(smiles=smiles, memory_size=4, sample_size=4)
        scoring = ComponentParameters(component_type=sf_enum.JACCARD_DISTANCE,
                                      name="jaccard_distance",
                                      weight=1.,
                                      smiles=["CONN", "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"],
                                      model_path="",
                                      specific_parameters={})
        scoringfunction = CustomSum(parameters=[scoring])
        self.inception_model = Inception(configuration=config, scoring_function=scoringfunction, prior=prior)

        self.inception_model.add(smiles, score, prior_likelihood)

    def tearDown(self):
        if os.path.isdir(self.log_path):
            shutil.rmtree(self.log_path)

    def test_empty_add(self):
        smiles = np.array(['CCC', 'CCCC', 'CC'])
        score = [0, 0.5, 1.]
        prior_likelihood = ts.tensor([0, 10, 100])
        self.inception_model.add(smiles, score, prior_likelihood)
        self.assertEqual(len(self.inception_model.memory), 3)

    def test_empty_add_2(self):
        smiles = np.array(['CCC', 'CCCC', 'CCCCC', 'COCNNC'])
        score = [0, 0.5, 0.5, 1]
        prior_likelihood = ts.tensor([0, 10, 10, 100])
        self.inception_model.add(smiles, score, prior_likelihood)
        self.assertEqual(len(self.inception_model.sample()[0]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[1]), self.inception_model.configuration.sample_size)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.sample_size)
        nt.assert_almost_equal(np.array([1, 0.5, 0.5, 0]), np.array(self.inception_model.memory['score'].values))
        nt.assert_almost_equal(np.array([100, 10, 10, 0]), np.array(self.inception_model.memory['likelihood'].values))

    def test_empty_eval_add_1(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        smiles = np.array(['CCC', 'CCCC', 'CC', 'COO'])
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
        nt.assert_almost_equal(np.array(self.inception_model.memory['score'].values), np.array([1, 1, 0.6667, 0.1250]),
                               4)
        self.assertEqual(len(np.array(self.inception_model.memory['likelihood'].values)), 4)
