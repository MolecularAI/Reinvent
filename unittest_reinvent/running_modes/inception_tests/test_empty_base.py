import os
import shutil
import unittest

import torch as ts

import running_modes.utils.general as utils_general
from reinvent_models.reinvent_core.models.model import Model
from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.reinforcement_learning.inception import Inception
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.function import CustomSum
from unittest_reinvent.fixtures.paths import PRIOR_PATH, MAIN_TEST_PATH
from unittest_reinvent.fixtures.test_data import METHOXYHYDRAZINE, ASPIRIN
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class TestInceptionEmptyBase(unittest.TestCase):

    def setUp(self):
        self.sf_enum = ScoringFunctionComponentNameEnum()
        utils_general.set_default_device_cuda()
        self.log_path = MAIN_TEST_PATH

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        smiles = []
        score = []
        prior_likelihood = ts.tensor([])

        prior = Model.load_from_file(PRIOR_PATH)
        config = InceptionConfiguration(smiles=smiles, memory_size=4, sample_size=4)
        scoring = ComponentParameters(component_type=self.sf_enum.JACCARD_DISTANCE,
                                      name=self.sf_enum.JACCARD_DISTANCE,
                                      weight=1.,
                                      smiles=[METHOXYHYDRAZINE, ASPIRIN],
                                      model_path="",
                                      specific_parameters={})
        scoring_function = CustomSum(parameters=[scoring])

        self.inception_model = Inception(configuration=config, scoring_function=scoring_function, prior=prior)
        self.inception_model.add(smiles, score, prior_likelihood)

    def tearDown(self):
        if os.path.isdir(self.log_path):
            shutil.rmtree(self.log_path)
