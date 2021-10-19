import os
import shutil
import unittest

import torch
from reinvent_chemistry.library_design.reaction_filters import ReactionFiltersEnum
from reinvent_chemistry.library_design.reaction_filters.reaction_filter_configruation import ReactionFilterConfiguration
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_scoring import ComponentParameters, ScoringFunctionComponentNameEnum, ScoringFunctionNameEnum
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.configurations import ReinforcementLoggerConfiguration, GeneralConfigurationEnvelope
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import \
    LearningStrategyConfiguration

from running_modes.reinforcement_learning.configurations.lib_invent_reinforcement_learning_configuration import \
    LibInventReinforcementLearningConfiguration
from running_modes.reinforcement_learning.configurations.lib_invent_scoring_strategy_configuration import \
    LibInventScoringStrategyConfiguration
from running_modes.reinforcement_learning.learning_strategy import LearningStrategyEnum
from running_modes.reinforcement_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.reinforcement_learning.lib_invent_reinforcement_learning import LibInventReinforcementLearning
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.reinforcement_learning.scoring_strategy.scoring_strategy import ScoringStrategy
from running_modes.reinforcement_learning.scoring_strategy.scoring_strategy_enum import ScoringStrategyEnum
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.paths import LIBINVENT_PRIOR_PATH, MAIN_TEST_PATH
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, IBUPROFEN, SCAFFOLD_SUZUKI
from unittest_reinvent.fixtures.utils import count_empty_files


class TestReinforcementLearningModel(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()

        model_regime = GenerativeModelRegimeEnum()
        self.sf_component_enum = ScoringFunctionComponentNameEnum()
        self.sf_enum = ScoringFunctionNameEnum()
        self.rf_enum = ReactionFiltersEnum()
        self.ss_enum = ScoringStrategyEnum()
        self.ls_enum = LearningStrategyEnum()
        model_type_enum = ModelTypeEnum()
        rt_enum = RunningModeEnum()

        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        log_config = ReinforcementLoggerConfiguration(logging_path=self.logging_path, recipient="local",
                                                      result_folder=self.workfolder)

        reactions = {"0": ["[*;$(c2aaaaa2),$(c2aaaa2):1]-!@[*;$(c2aaaaa2),$(c2aaaa2):2]>>[*:1][*].[*:2][*]"]}
        smiles = [CELECOXIB, ASPIRIN, IBUPROFEN]

        reaction_filter_config = ReactionFilterConfiguration(type=self.rf_enum.SELECTIVE, reactions=reactions)
        diversity_filter_parameters = DiversityFilterParameters(name="NoFilterWithPenalty")
        sf_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 specific_parameters={"smiles":smiles},
                                                 component_type=self.sf_component_enum.TANIMOTO_SIMILARITY))

        scoring_function_parameters = ScoringFunctionParameters(name=self.sf_enum.CUSTOM_SUM,parameters=[sf_parameters])
        learning_strategy_config = LearningStrategyConfiguration(name=self.ls_enum.DAP, parameters={"sigma": 100})
        scoring_strategy_config = LibInventScoringStrategyConfiguration(name=self.ss_enum.LIB_INVENT,
                                                                        reaction_filter=reaction_filter_config,
                                                               diversity_filter=diversity_filter_parameters,
                                                               scoring_function=scoring_function_parameters)
        config = LibInventReinforcementLearningConfiguration(actor=LIBINVENT_PRIOR_PATH, critic=LIBINVENT_PRIOR_PATH,
                                                    scaffolds=[SCAFFOLD_SUZUKI],
                                                    learning_strategy=learning_strategy_config,
                                                    scoring_strategy=scoring_strategy_config, n_steps=2, batch_size=4)
        critic_config = ModelConfiguration(model_type_enum.LIB_INVENT, model_regime.INFERENCE, config.critic)
        actor_config = ModelConfiguration(model_type_enum.LIB_INVENT, model_regime.TRAINING, config.actor)
        critic = GenerativeModel(critic_config)
        actor = GenerativeModel(actor_config)
        config_envelope = GeneralConfigurationEnvelope({}, {}, rt_enum.REINFORCEMENT_LEARNING, "3",
                                                       model_type=model_type_enum.LIB_INVENT)
        logger = ReinforcementLogger(config_envelope, log_config)

        optimizer = torch.optim.Adam(actor.get_network_parameters(), lr=config.learning_rate)
        diversity_filter = DiversityFilter(config.scoring_strategy.diversity_filter)
        learning_strategy = LearningStrategy(critic, optimizer, config.learning_strategy, logger)
        scoring_strategy = ScoringStrategy(config.scoring_strategy, diversity_filter, logger)

        self.runner = LibInventReinforcementLearning(critic=critic, actor=actor, configuration=config,
                                                     learning_strategy=learning_strategy,
                                                     scoring_strategy=scoring_strategy, logger=logger)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _check_log_directory(self):
        self.assertEqual(os.path.isdir(self.logging_path), True)
        self.assertEqual(len(os.listdir(self.logging_path)) >= 1, True)
        self.assertEqual(count_empty_files(self.logging_path), 0)

    def test_reinforcement_learning_model(self):
        self.runner.run()
        self._check_log_directory()
