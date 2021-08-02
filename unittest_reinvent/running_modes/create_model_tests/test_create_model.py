import os
import shutil
import unittest

from running_modes.configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.create_model.create_model_configuration import CreateModelConfiguration
from running_modes.create_model.create_model import CreateModelRunner
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, SMILES_SET_PATH
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.enums.logging_mode_enum import LoggingModeEnum

class TestCreateModel(unittest.TestCase):

    def setUp(self):
        self.rm_enums = RunningModeEnum()
        self.lm_enum = LoggingModeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "new_model.ckpt")
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.config = CreateModelConfiguration(input_smiles_path=SMILES_SET_PATH, output_model_path=self.output_file, standardize=True)
        log_conf = CreateModelLoggerConfiguration(recipient=self.lm_enum.LOCAL,
                                                  logging_path=f"{MAIN_TEST_PATH}/log",
                                                  job_name="create_model_test",
                                                  job_id="1")
        self.configuration_envelope = GeneralConfigurationEnvelope(parameters=vars(self.config),
                                                                   logging=vars(log_conf),
                                                                   run_type=self.rm_enums.VALIDATION,
                                                                   version="2.0")

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_save_model_1(self):
        runner = CreateModelRunner(self.configuration_envelope, self.config)
        runner.run()
        self.assertEqual(os.path.isfile(self.output_file), True)

    def test_save_model_2(self):
        runner = CreateModelRunner(self.configuration_envelope, self.config)
        model = runner.run()
        self.assertEqual(model.max_sequence_length, self.config.max_sequence_length)
        self.assertEqual(len(model.vocabulary.tokens()), 29)

    def test_save_model_3(self):
        self.config.standardize = False
        runner = CreateModelRunner(self.configuration_envelope, self.config)
        model = runner.run()
        self.assertEqual(model.max_sequence_length, self.config.max_sequence_length)
        self.assertEqual(len(model.vocabulary.tokens()), 30)
