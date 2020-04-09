import os
import shutil
import unittest

import numpy as np
import torch as ts

import utils.general as utils_general
from models.model import Model
from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.reinforcement_learning.inception import Inception
from scoring.component_parameters import ComponentParameters
from scoring.function import CustomSum
from unittest_reinvent.fixtures.paths import RANDOM_PRIOR_PATH, MAIN_TEST_PATH
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class Test_inception_model_add(unittest.TestCase):
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
        scoringfunction = CustomSum(parameters=[scoring])
        self.inception_model = Inception(configuration=config, scoring_function=scoringfunction, prior=prior)

        self.inception_model.add(smiles, score, prior_likelihood)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.log_path):
            shutil.rmtree(self.log_path)

    def test_inception_model_1(self):
        self.assertEqual(len(self.inception_model.memory), 3)

    def test_inception_model_2(self):
        smiles = np.array(['CCC', 'CCCC', 'CC'])
        score = [0, 0.5, 1.]
        prior_likelihood = ts.tensor([0, 10, 100])
        self.inception_model.add(smiles, score, prior_likelihood)
        self.assertEqual(len(self.inception_model.memory), 4)

    def test_inception_model_3(self):
        smiles = np.array(['CCC', 'CCCC', 'CO'])
        score = [0, 0.5, 1.]
        prior_likelihood = ts.tensor([0, 10, 100])
        self.inception_model.add(smiles, score, prior_likelihood)
        self.assertEqual(len(self.inception_model.memory), 4)

    def test_inception_model_4(self):
        exp_seqs, exp_score, exp_prior_likelihood = self.inception_model.sample()
        self.assertEqual(len(exp_seqs), 4)

    def test_inception_model_5(self):
        exp_seqs, exp_score, exp_prior_likelihood = self.inception_model.sample()
        self.assertEqual(len(exp_seqs), self.inception_model.configuration.memory_size)
        self.assertEqual(len(exp_score), self.inception_model.configuration.sample_size)
        self.assertEqual(len(exp_prior_likelihood), len(exp_seqs))
        self.assertEqual(len(exp_prior_likelihood), len(exp_score))

    def test_inception_model_6(self):
        len0 = len(self.inception_model.sample()[0])
        len1 = len(self.inception_model.sample()[1])
        len2 = len(self.inception_model.sample()[2])
        smiles = np.array(['CCC', 'CCCC'])
        score = [0, 0.5]
        prior_likelihood = ts.tensor([0, 10])
        self.inception_model.add(smiles, score, prior_likelihood)
        self.assertEqual(len(self.inception_model.sample()[0]), len0)
        self.assertEqual(len(self.inception_model.sample()[0]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[1]), len1)
        self.assertEqual(len(self.inception_model.sample()[1]), self.inception_model.configuration.sample_size)
        self.assertEqual(len(self.inception_model.sample()[2]), len2)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.sample_size)
