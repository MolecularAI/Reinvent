import os
import shutil
import unittest

from reinvent_chemistry import TransformationTokens

from running_modes.configurations import LinkInventCreateModelConfiguration, GeneralConfigurationEnvelope
from running_modes.configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration
from running_modes.create_model import LinkInventCreateModelRunner
from running_modes.create_model.logging.local_create_model_logger import LocalCreateModelLogger
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, SMILES_SET_LINK_INVENT_PATH
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum


class TestLinkInventCreateModel(unittest.TestCase):

    def setUp(self):
        self.rm_enums = RunningModeEnum()
        self.lm_enum = LoggingModeEnum()
        self.mt_enum = ModelTypeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "new_model.empty")
        os.makedirs(self.workfolder, exist_ok=True)
        self.config = LinkInventCreateModelConfiguration(input_smiles_path=SMILES_SET_LINK_INVENT_PATH,
                                                         output_model_path=self.output_file)
        log_conf = CreateModelLoggerConfiguration(recipient=self.lm_enum.LOCAL,
                                                  logging_path=os.path.join(MAIN_TEST_PATH, "log"),
                                                  job_name="link_invent_create_model_test",
                                                  job_id="1")
        self.configuration_envelope = GeneralConfigurationEnvelope(parameters=vars(self.config),
                                                                   logging=vars(log_conf),
                                                                   run_type=self.rm_enums.CREATE_MODEL,
                                                                   model_type=self.mt_enum.LINK_INVENT,
                                                                   version="3.0")
        self.logger = LocalCreateModelLogger(self.configuration_envelope)

        runner = LinkInventCreateModelRunner(self.config, self.logger)
        self.model = runner.run()

        self.tokens = TransformationTokens()

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_save_model(self):
        self.assertTrue(os.path.isfile(self.config.output_model_path))

    def test_correct_model_properties(self):
        self.assertEqual(self.model.max_sequence_length, self.config.max_sequence_length)
        self.assertEqual(len(self.model.vocabulary.input), 31)
        self.assertEqual(len(self.model.vocabulary.target), 30)

    def test_correct_assignment_of_decoder_encoder(self):
        self.assertIn(self.tokens.ATTACHMENT_SEPARATOR_TOKEN, self.model.vocabulary.input.vocabulary.tokens())
        self.assertNotIn(self.tokens.ATTACHMENT_SEPARATOR_TOKEN, self.model.vocabulary.target.vocabulary.tokens())

    def test_padding_voc(self):
        self.assertEqual(self.model.vocabulary.input.decode([0]), '<pad>')
        self.assertEqual(self.model.vocabulary.target.decode([0]), '<pad>')

        self.assertEqual(self.model.vocabulary.input.vocabulary.encode(['<pad>']), 0)
        self.assertEqual(self.model.vocabulary.target.vocabulary.encode(['<pad>']), 0)
