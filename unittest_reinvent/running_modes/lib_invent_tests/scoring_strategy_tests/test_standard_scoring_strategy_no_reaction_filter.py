import os
import shutil
import unittest

from reinvent_chemistry.library_design.reaction_filters.reaction_filter_configruation import ReactionFilterConfiguration
from reinvent_chemistry.library_design.reaction_filters.reaction_filter_enum import ReactionFiltersEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums.scoring_function_enum import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.configurations import ReinforcementLoggerConfiguration, GeneralConfigurationEnvelope
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.reinforcement_learning.configurations.lib_invent_scoring_strategy_configuration import \
    LibInventScoringStrategyConfiguration
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.reinforcement_learning.scoring_strategy.lib_invent_scoring_strategy import LibInventScoringStrategy
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, BENZENE
from unittest_reinvent.fixtures.utils import count_empty_files


class TestStandardScoringStrategyNoReactionFilter(unittest.TestCase):

    def setUp(self):
        self.sf_component_enum = ScoringFunctionComponentNameEnum
        self.sf_enum = ScoringFunctionNameEnum
        self.enum = ReactionFiltersEnum()
        rt_enum = RunningModeEnum()
        mt_enum = ModelTypeEnum()

        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        smiles = [CELECOXIB, ASPIRIN, BENZENE]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 specific_parameters={"smiles":smiles},
                                                 component_type=self.sf_component_enum.TANIMOTO_SIMILARITY))
        sf_parameters = ScoringFunctionParameters(name=self.sf_enum.CUSTOM_SUM, parameters=[ts_parameters])
        log_config = ReinforcementLoggerConfiguration(logging_path=self.logging_path,
                                                      job_name="lib_invent reinforcement logger",
                                                      recipient="local", result_folder=self.workfolder)

        df_parameters = DiversityFilterParameters(name="NoFilter")
        reaction_filter = ReactionFilterConfiguration(self.enum.SELECTIVE, {})

        scoring_strategy_config = LibInventScoringStrategyConfiguration(reaction_filter=reaction_filter,
                                                               diversity_filter=df_parameters,
                                                               scoring_function=sf_parameters,
                                                               name="test")
        config_envelope = GeneralConfigurationEnvelope({}, {}, rt_enum.REINFORCEMENT_LEARNING, "3",
                                                       model_type=mt_enum.LINK_INVENT)
        logger = ReinforcementLogger(config_envelope, log_config)
        diversity_filter = DiversityFilter(df_parameters)

        self.standard_scoring = LibInventScoringStrategy(scoring_strategy_config, diversity_filter, logger)

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
