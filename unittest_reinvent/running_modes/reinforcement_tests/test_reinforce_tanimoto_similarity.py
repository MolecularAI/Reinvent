import os
import shutil
import unittest

from reinvent_models.reinvent_core.models.model import Model

from running_modes.configurations.logging.reinforcement_log_configuration import ReinforcementLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.configurations.reinforcement_learning.reinforcement_learning_components import \
    ReinforcementLearningComponents
from running_modes.configurations.reinforcement_learning.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.reinforcement_runner import ReinforcementRunner
from running_modes.utils import set_default_device_cuda
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import DiversityFilterParameters

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFuncionParameters
from reinvent_scoring.scoring.enums.scoring_function_enum import ScoringFunctionNameEnum
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum

from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, PRIOR_PATH
from unittest_reinvent.fixtures.test_data import PROPANE, ASPIRIN
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.enums.scaffold_filter_enum import ScaffoldFilterEnum


class TestReinforceTanimotoSimilarity(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        lm_enum = LoggingModeEnum()
        run_mode_enum = RunningModeEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        filter_enum = ScaffoldFilterEnum()
        self.workfolder = MAIN_TEST_PATH
        smiles = [PROPANE, ASPIRIN]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 smiles=smiles, model_path="", specific_parameters={},
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY))

        sf_parameters = ScoringFuncionParameters(name=sf_enum.CUSTOM_SUM, parameters=[ts_parameters])
        scoring_function = ScoringFunctionFactory(sf_parameters)

        logging = ReinforcementLoggerConfiguration(recipient=lm_enum.LOCAL,
                                                   logging_path=f"{self.workfolder}/log", resultdir=self.workfolder,
                                                   logging_frequency=0, job_name="unit_test_job")

        rl_config = ReinforcementLearningConfiguration(prior=PRIOR_PATH, agent=PRIOR_PATH,
                                                       n_steps=3)

        scaffold_parameters = DiversityFilterParameters(filter_enum.IDENTICAL_MURCKO_SCAFFOLD, 0.05, 25, 0.4)
        diversity_filter = self._setup_scaffold_filter(scaffold_parameters)
        inception_config = InceptionConfiguration(smiles, 100, 10)
        inception = Inception(inception_config, scoring_function, Model.load_from_file(rl_config.prior))
        parameters = ReinforcementLearningComponents(scoring_function=sf_parameters,
                                                     diversity_filter=scaffold_parameters,
                                                     reinforcement_learning=rl_config,
                                                     inception=inception_config)
        configuration = GeneralConfigurationEnvelope(parameters=vars(parameters), logging=vars(logging),
                                                     run_type=run_mode_enum.REINFORCEMENT_LEARNING, version="2.0")

        self.runner = ReinforcementRunner(configuration, config=rl_config,
                                          diversity_filter=diversity_filter, scoring_function=scoring_function,
                                          inception=inception)

    @staticmethod
    def _setup_scaffold_filter(diversity_filter_parameters):
        diversity_filter = DiversityFilter(diversity_filter_parameters)
        return diversity_filter

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_reinforcement_with_similarity_run_1(self):
        self.runner.run()
        self.assertEqual(os.path.isdir(f"{self.workfolder}/log"), True)
