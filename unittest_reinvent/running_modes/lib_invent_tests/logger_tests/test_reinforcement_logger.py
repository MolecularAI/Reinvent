import unittest
import os
import shutil

from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters

from running_modes.lib_invent.configurations.log_configuration import LogConfiguration
from running_modes.lib_invent.logging.reinforcement_logger import ReinforcementLogger
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from reinvent_scoring.scoring.diversity_filters.lib_invent import NoFilter
from unittest_reinvent.fixtures.utils import count_empty_files


class TestReinforcementLogger(unittest.TestCase):

    def setUp(self):
        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        log_config = LogConfiguration(logging_path=self.logging_path,
                                      job_name="lib_invent reinforcement logger")
        diversity_parameters = DiversityFilterParameters(name="test")
        self.diversity_filter = NoFilter(diversity_parameters)
        self.logger = ReinforcementLogger(log_config)

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
