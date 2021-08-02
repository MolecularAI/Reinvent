import unittest
import os
import shutil

from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters

from running_modes.lib_invent.configurations.log_configuration import LogConfiguration
from running_modes.lib_invent.logging.reinforcement_logger import ReinforcementLogger
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from reinvent_scoring.scoring.diversity_filters.lib_invent import NoFilter
from unittest_reinvent.fixtures.utils import count_empty_files

from reinvent_scoring.scoring.enums.scoring_function_enum import ScoringFunctionNameEnum
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFuncionParameters
from running_modes.lib_invent.scoring_strategy.standard_strategy import StandardScoringStrategy
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, BENZENE
from reinvent_chemistry.library_design.reaction_filters.reaction_filter_configruation import ReactionFilterConfiguration
from running_modes.lib_invent.configurations.scoring_strategy_configuration import ScoringStrategyConfiguration
from reinvent_chemistry.library_design.reaction_filters.reaction_filter_enum import ReactionFiltersEnum


class TestStandardScoringStrategyNoReactionFilter(unittest.TestCase):

    def setUp(self):
        self.sf_component_enum = ScoringFunctionComponentNameEnum
        self.sf_enum = ScoringFunctionNameEnum
        self.enum = ReactionFiltersEnum()

        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        smiles = [CELECOXIB, ASPIRIN, BENZENE]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 smiles=smiles, model_path="", specific_parameters={},
                                                 component_type=self.sf_component_enum.TANIMOTO_SIMILARITY))
        sf_parameters = ScoringFuncionParameters(name=self.sf_enum.CUSTOM_SUM, parameters=[ts_parameters])
        log_config = LogConfiguration(logging_path=self.logging_path,
                                      job_name="lib_invent reinforcement logger")

        df_parameters = DiversityFilterParameters(name="NoFilter")
        reaction_filter = ReactionFilterConfiguration(self.enum.SELECTIVE, {})
        scoring_strategy_config = ScoringStrategyConfiguration(reaction_filter=reaction_filter,
                                                               diversity_filter=df_parameters,
                                                               scoring_function=sf_parameters,
                                                               name="test")
        logger = ReinforcementLogger(log_config)

        self.standard_scoring = StandardScoringStrategy(scoring_strategy_config, logger)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _check_log_directory(self):
        self.assertEqual(os.path.isdir(self.logging_path), True)
        self.assertEqual(len(os.listdir(self.logging_path)) >= 1, True)
        self.assertEqual(count_empty_files(self.logging_path), 0)

    def test_standard_scoring_strategy_log_directory(self):
        self.standard_scoring.save_filter_memory()
        self._check_log_directory()
