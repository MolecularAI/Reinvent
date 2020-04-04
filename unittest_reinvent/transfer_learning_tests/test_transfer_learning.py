import os
import shutil
import unittest

import utils as utils_general
import models.model as reinvent_model
from running_modes.configurations.logging.transfer_learning_log_configuration import TransferLearningLoggerConfig
from running_modes.transfer_learning.logging.local_transfer_learning_logger import LocalTransferLearningLogger
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.transfer_learning.adaptive_learning_rate_configuration import AdaptiveLearningRateConfiguration
from running_modes.configurations.transfer_learning.transfer_learning_configuration import TransferLearningConfiguration
from running_modes.transfer_learning.adaptive_learning_rate import AdaptiveLearningRate
from running_modes.transfer_learning.transfer_learning_runner import TransferLearningRunner
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, SMILES_SET_PATH, RANDOM_PRIOR_PATH
from utils.enums.logging_mode_enum import LoggingModeEnum
from utils.enums.running_mode_enum import RunningModeEnum


class Test_transfer_learning_no_validation_set(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        utils_general.set_default_device_cuda()
        lm_enum = LoggingModeEnum()
        rm_enum = RunningModeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "new_model.ckpt")
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        logdir = os.path.join(self.workfolder, "test_log")
        logconfig = TransferLearningLoggerConfig(logging_path=logdir, sender="http://10.59.162.10:8081",
                                                 recipient=lm_enum.LOCAL, job_name="test_job", use_weights=0)
        self.alr_config = AdaptiveLearningRateConfiguration()
        self.parameters = TransferLearningConfiguration(input_model_path=RANDOM_PRIOR_PATH,
                                                        output_model_path=self.output_file,
                                                        input_smiles_path=SMILES_SET_PATH,
                                                        validation_smiles_path=None,
                                                        num_epochs=3, adaptive_lr_config=self.alr_config)
        self.config = GeneralConfigurationEnvelope(parameters=vars(self.parameters), logging=vars(logconfig),
                                                   run_type=rm_enum.TRANSFER_LEARNING, version="2.0")
        self.stats_logger = LocalTransferLearningLogger(configuration=self.config)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_transfer_learning_no_validation(self):

        model = reinvent_model.Model.load_from_file(self.parameters.input_model_path)
        adaptive_learning_rate = AdaptiveLearningRate(model=model, main_config=self.config,
                                                      configuration=self.alr_config)
        runner = TransferLearningRunner(model=model, config=self.parameters,
                                        adaptive_learning_rate=adaptive_learning_rate)
        runner.run()
        self.assertEqual(os.path.isfile(self.output_file), True)
