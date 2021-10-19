import unittest
import os
import shutil

from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters

from running_modes.configurations import ReinforcementLoggerConfiguration, GeneralConfigurationEnvelope
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from reinvent_scoring.scoring.diversity_filters.lib_invent import NoFilter
from unittest_reinvent.fixtures.utils import count_empty_files


class TestReinforcementLogger(unittest.TestCase):

    def setUp(self):
        model_type_enum = ModelTypeEnum()
        rt_enum = RunningModeEnum()
        logging_mode_enum = LoggingModeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        log_config = ReinforcementLoggerConfiguration(logging_path=self.logging_path, recipient=logging_mode_enum.LOCAL,
                                                      result_folder=self.workfolder)
        diversity_parameters = DiversityFilterParameters(name="NoFilter")
        self.diversity_filter = NoFilter(diversity_parameters)
        config_envelope = GeneralConfigurationEnvelope({}, {}, rt_enum.REINFORCEMENT_LEARNING, "3",
                                                       model_type=model_type_enum.LIB_INVENT)
        self.logger = ReinforcementLogger(config_envelope, log_config)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _check_log_directory(self):
        self.assertEqual(os.path.isdir(self.logging_path), True)
        self.assertEqual(len(os.listdir(self.logging_path)) >= 1, True)
        self.assertEqual(count_empty_files(self.logging_path), 0)

    def test_reinforcement_logger(self):
        self.logger.save_filter_memory(self.diversity_filter)
        self._check_log_directory()
