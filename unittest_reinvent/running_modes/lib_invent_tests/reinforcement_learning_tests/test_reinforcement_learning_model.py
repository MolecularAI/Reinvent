import os
import shutil
import unittest

from reinvent_chemistry.library_design.reaction_filters import ReactionFiltersEnum
from reinvent_chemistry.library_design.reaction_filters.reaction_filter_configruation import ReactionFilterConfiguration
from reinvent_scoring import ComponentParameters, ScoringFunctionComponentNameEnum, ScoringFunctionNameEnum

from running_modes.lib_invent.configurations.learning_strategy_configuration import LearningStrategyConfiguration
from running_modes.lib_invent.configurations.log_configuration import LogConfiguration
from running_modes.lib_invent.configurations.scoring_strategy_configuration import ScoringStrategyConfiguration
from running_modes.lib_invent.learning_strategy.learning_strategy_enum import LearningStrategyEnum
from running_modes.lib_invent.lib_invent_reinforcement_learning import LibInventReinforcementLearning
from running_modes.lib_invent.logging.reinforcement_logger import ReinforcementLogger
from reinvent_models.lib_invent.models.model import DecoratorModel
from running_modes.lib_invent.configurations.reinforcement_learning_configuration import \
    ReinforcementLearningConfiguration
from running_modes.lib_invent.scoring_strategy.scoring_strategy_enum import ScoringStrategyEnum
from unittest_reinvent.fixtures.paths import LIBINVENT_PRIOR_PATH, MAIN_TEST_PATH
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, IBUPROFEN, SCAFFOLD_SUZUKI
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFuncionParameters
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum

from unittest_reinvent.fixtures.utils import count_empty_files


class TestReinforcementLearningModel(unittest.TestCase):

    def setUp(self):
        model_regime = GenerativeModelRegimeEnum()
        self.sf_component_enum = ScoringFunctionComponentNameEnum()
        self.sf_enum = ScoringFunctionNameEnum()
        self.rf_enum = ReactionFiltersEnum()
        self.ss_enum = ScoringStrategyEnum()
        self.ls_enum = LearningStrategyEnum()

        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        log_config = LogConfiguration(logging_path=self.logging_path,
                                      job_name="lib_invent reinforcement logger")

        reactions = {
            "0": ["[*;$(c2aaaaa2),$(c2aaaa2):1]-!@[*;$(c2aaaaa2),$(c2aaaa2):2]>>[*:1][*].[*:2][*]"],
        }
        smiles = [CELECOXIB, ASPIRIN, IBUPROFEN]

        reaction_filter_config = ReactionFilterConfiguration(type=self.rf_enum.SELECTIVE,
                                                             reactions=reactions)
        diversity_filter_parameters = DiversityFilterParameters(name="NoFilterWithPenalty")
        tanimoto_similarity_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                                  smiles=smiles, model_path="", specific_parameters={},
                                                                  component_type=self.sf_component_enum.TANIMOTO_SIMILARITY))
        scoring_function_parameters = ScoringFuncionParameters(name=self.sf_enum.CUSTOM_SUM,
                                                               parameters=[tanimoto_similarity_parameters])
        learning_strategy_config = LearningStrategyConfiguration(name=self.ls_enum.DAP, parameters={"sigma": 100})
        scoring_strategy_config = ScoringStrategyConfiguration(name=self.ss_enum.STANDARD, reaction_filter=reaction_filter_config,
                                                               diversity_filter=diversity_filter_parameters,
                                                               scoring_function=scoring_function_parameters)
        config = ReinforcementLearningConfiguration(actor=LIBINVENT_PRIOR_PATH, critic=LIBINVENT_PRIOR_PATH,
                                                    scaffolds=[SCAFFOLD_SUZUKI],
                                                    learning_strategy=learning_strategy_config,
                                                    scoring_strategy=scoring_strategy_config, n_steps=2, batch_size=4)

        critic = DecoratorModel.load_from_file(config.critic, mode=model_regime.INFERENCE)
        actor = DecoratorModel.load_from_file(config.actor, mode=model_regime.TRAINING)
        logger = ReinforcementLogger(log_config)
        self.runner = LibInventReinforcementLearning(critic=critic, actor=actor, configuration=config, logger=logger)

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
