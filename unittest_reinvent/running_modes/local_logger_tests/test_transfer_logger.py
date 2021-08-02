import os
import shutil
import unittest
import numpy as np

from running_modes.transfer_learning.logging.local_transfer_learning_logger import LocalTransferLearningLogger
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, PRIOR_PATH, SMILES_SET_PATH
from unittest_reinvent.fixtures.test_data import ASPIRIN, METAMIZOLE, PROPANE, IBUPROFEN, CELECOXIB, LIKELIHOODLIST, \
    REP_LIKELIHOOD, REP_SMILES_LIST, INVALID_SMILES_LIST
from unittest_reinvent.fixtures.utils import count_empty_files
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.configurations.transfer_learning.transfer_learning_configuration import TransferLearningConfiguration
from running_modes.configurations.transfer_learning.adaptive_learning_rate_configuration import \
    AdaptiveLearningRateConfiguration
from running_modes.configurations.logging.transfer_learning_log_configuration import TransferLearningLoggerConfig
from running_modes.enums.running_mode_enum import RunningModeEnum


class TestTransferLogger(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        rm_enum = RunningModeEnum()
        lm_enum = LoggingModeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "new_model.ckpt")
        self.logging_path = f"{self.workfolder}/TRlog"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        logconfig = TransferLearningLoggerConfig(logging_path=self.logging_path,
                                                 recipient=lm_enum.LOCAL, job_name="test_job", use_weights=0)
        self.alr_config = AdaptiveLearningRateConfiguration()
        self.parameters = TransferLearningConfiguration(input_model_path=PRIOR_PATH,
                                                        output_model_path=self.output_file,
                                                        input_smiles_path=SMILES_SET_PATH,
                                                        num_epochs=3, adaptive_lr_config=self.alr_config)
        self.config = GeneralConfigurationEnvelope(parameters=vars(self.parameters), logging=vars(logconfig),
                                                   run_type=rm_enum.TRANSFER_LEARNING, version="2.0")
        self.stats_logger = LocalTransferLearningLogger(configuration=self.config)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _check_log_directory(self):
        self.assertEqual(os.path.isdir(self.logging_path), True)
        self.assertEqual(len(os.listdir(self.logging_path)) >= 1, True)
        self.assertEqual(count_empty_files(self.logging_path), 0)

    def test_transfer_with_only_invalids(self):
        smiles = INVALID_SMILES_LIST
        self.stats_logger.log_timestep(self.alr_config.start, 0, smiles,
                                       LIKELIHOODLIST, None,
                                       LIKELIHOODLIST, {"jsd": 5}, np.array([5]), PRIOR_PATH, None)
        self._check_log_directory()

    def test_transfer_logging_with_some_valids(self):
        smiles = [ASPIRIN, METAMIZOLE, PROPANE, IBUPROFEN, CELECOXIB]
        self.stats_logger.log_timestep(self.alr_config.start, 0, smiles, LIKELIHOODLIST[:5], None,
                                       LIKELIHOODLIST[:5], {"jsd": 5}, np.array([5]), PRIOR_PATH, None)
        self._check_log_directory()

    def test_transfer_logging_with_duplicate_valids(self):
        smiles = REP_SMILES_LIST
        self.stats_logger.log_timestep(self.alr_config.start, 0, smiles,
                                       REP_LIKELIHOOD, None,
                                       REP_LIKELIHOOD, {"jsd": 5}, np.array([5]), PRIOR_PATH, None)
        self._check_log_directory()
