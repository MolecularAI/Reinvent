import os
import shutil
import unittest

from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters
from reinvent_scoring.scoring.enums.diversity_filter_enum import DiversityFilterEnum
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums.scoring_function_enum import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.configurations import ReinforcementLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.configurations.reinforcement_learning.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.reinforcement_learning import CoreReinforcementRunner, Inception
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, PRIOR_PATH
from unittest_reinvent.fixtures.test_data import PROPANE, ASPIRIN


class TestReinforceTanimotoSimilarity(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        lm_enum = LoggingModeEnum()
        run_mode_enum = RunningModeEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        filter_enum = DiversityFilterEnum()
        model_regime = GenerativeModelRegimeEnum()
        model_type_enum = ModelTypeEnum()
        self.workfolder = MAIN_TEST_PATH

        smiles = [PROPANE, ASPIRIN]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 specific_parameters={"smiles": smiles},
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY))

        sf_parameters = ScoringFunctionParameters(name=sf_enum.CUSTOM_SUM, parameters=[ts_parameters])
        scoring_function = ScoringFunctionFactory(sf_parameters)
        scaffold_parameters = DiversityFilterParameters(filter_enum.IDENTICAL_MURCKO_SCAFFOLD, 0.05, 25, 0.4)

        prior_config = ModelConfiguration(model_type_enum.DEFAULT, model_regime.INFERENCE, PRIOR_PATH)
        actor_config = ModelConfiguration(model_type_enum.DEFAULT, model_regime.TRAINING, PRIOR_PATH)
        prior = GenerativeModel(prior_config)
        actor = GenerativeModel(actor_config)
        inception_config = InceptionConfiguration(smiles, 100, 10)
        inception = Inception(inception_config, scoring_function, prior)
        log_config = ReinforcementLoggerConfiguration(recipient=lm_enum.LOCAL,
                                                      logging_path=f"{self.workfolder}/log", result_folder=self.workfolder,
                                                      logging_frequency=0, job_name="unit_test_job")
        configuration = GeneralConfigurationEnvelope(parameters={}, logging={},
                                                     run_type=run_mode_enum.REINFORCEMENT_LEARNING, version="2.0")
        logger = ReinforcementLogger(configuration, log_config)

        diversity_filter = self._setup_scaffold_filter(scaffold_parameters)
        config = ReinforcementLearningConfiguration(prior=PRIOR_PATH, agent=PRIOR_PATH, n_steps=3)

        self.runner = CoreReinforcementRunner(prior, actor, config, scoring_function, diversity_filter, inception, logger)

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
