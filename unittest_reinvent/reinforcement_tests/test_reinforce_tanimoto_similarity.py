import os
import shutil
import unittest

import utils as utils_general
from models.model import Model
from running_modes.configurations.logging.reinforcement_log_configuration import ReinforcementLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.configurations.reinforcement_learning.reinforcement_learning_components import \
    ReinforcementLearningComponents
from running_modes.configurations.reinforcement_learning.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.reinforcement_runner import ReinforcementRunner
from scaffold.scaffold_filter_factory import ScaffoldFilterFactory
from scaffold.scaffold_parameters import ScaffoldParameters
from scoring.component_parameters import ComponentParameters
from scoring.scoring_function_factory import ScoringFunctionFactory
from scoring.scoring_function_parameters import ScoringFuncionParameters
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, RANDOM_PRIOR_PATH
from utils.enums.logging_mode_enum import LoggingModeEnum
from utils.enums.running_mode_enum import RunningModeEnum
from utils.enums.scaffold_filter_enum import ScaffoldFilterEnum
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.enums.scoring_function_enum import ScoringFunctionNameEnum


class Test_reinforce_tanimoto_similarity(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        utils_general.set_default_device_cuda()
        lm_enum = LoggingModeEnum()
        run_mode_enum = RunningModeEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        filter_enum = ScaffoldFilterEnum()
        self.workfolder = MAIN_TEST_PATH
        smiles = ["CCC", "c1ccccc1CCC"]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 smiles=smiles, model_path="", specific_parameters={},
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY))

        sf_parameters = ScoringFuncionParameters(name=sf_enum.CUSTOM_SUM, parameters=[ts_parameters])
        scoring_function = ScoringFunctionFactory(sf_parameters)

        logging = ReinforcementLoggerConfiguration(sender="1", recipient=lm_enum.LOCAL,
                                                   logging_path=f"{self.workfolder}/log", resultdir=self.workfolder,
                                                   logging_frequency=0, job_name="unit_test_job")

        rl_config = ReinforcementLearningConfiguration(prior=RANDOM_PRIOR_PATH, agent=RANDOM_PRIOR_PATH,
                                                       n_steps=3)

        scaffold_parameters = ScaffoldParameters(filter_enum.IDENTICAL_MURCKO_SCAFFOLD, 0.05, 25, 0.4)
        scaffold_filter = self._setup_scaffold_filter(scaffold_parameters)
        inception_config = InceptionConfiguration(smiles, 100, 10)
        inception = Inception(inception_config, scoring_function, Model.load_from_file(rl_config.prior))
        parameters = ReinforcementLearningComponents(scoring_function=vars(scoring_function),
                                                     diversity_filter=vars(scaffold_filter),
                                                     reinforcement_learning=vars(rl_config),
                                                     inception=vars(inception_config))
        configuration = GeneralConfigurationEnvelope(parameters=vars(parameters), logging=vars(logging),
                                                     run_type=run_mode_enum.REINFORCEMENT_LEARNING, version="2.0")

        self.runner = ReinforcementRunner(configuration, config=rl_config,
                                          scaffold_filter=scaffold_filter, scoring_function=scoring_function,
                                          inception=inception)

    @staticmethod
    def _setup_scaffold_filter(scaffold_parameters):
        scaffold_factory = ScaffoldFilterFactory()
        scaffold = scaffold_factory.load_scaffold_filter(scaffold_parameters)
        return scaffold

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_reinforcement_with_similarity_run_1(self):
        self.runner.run()
        self.assertEqual(os.path.isdir(f"{self.workfolder}/log"), True)
