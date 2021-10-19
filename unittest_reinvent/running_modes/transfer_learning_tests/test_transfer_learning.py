import os
import shutil
import unittest

import reinvent_models.reinvent_core.models.model as reinvent_model
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.transfer_learning_log_configuration import TransferLearningLoggerConfig
from running_modes.configurations.transfer_learning.adaptive_learning_rate_configuration import \
    AdaptiveLearningRateConfiguration
from running_modes.configurations.transfer_learning.transfer_learning_configuration import TransferLearningConfiguration
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.transfer_learning.logging.transfer_learning_logger import TransferLearningLogger
from running_modes.transfer_learning.transfer_learning_runner import TransferLearningRunner
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, SMILES_SET_PATH, PRIOR_PATH


class TestTransferLearning(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        lm_enum = LoggingModeEnum()
        rm_enum = RunningModeEnum()
        standardization_filters = {"name": "default", "parameters": {"remove_long_side_chains": False}}
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "new_model.ckpt")
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.num_epochs = 3
        logdir = os.path.join(self.workfolder, "test_log")
        logconfig = TransferLearningLoggerConfig(logging_path=logdir, recipient=lm_enum.LOCAL,
                                                 job_name="test_job", use_weights=False)
        self.alr_config = AdaptiveLearningRateConfiguration()
        self.parameters = TransferLearningConfiguration(input_model_path=PRIOR_PATH,
                                                        output_model_path=self.output_file,
                                                        input_smiles_path=SMILES_SET_PATH,
                                                        validation_smiles_path=None,
                                                        num_epochs=self.num_epochs,
                                                        adaptive_lr_config=self.alr_config,
                                                        standardize=True,
                                                        standardization_filters=[standardization_filters],
                                                        validate_model_vocabulary=True)
        self.config = GeneralConfigurationEnvelope(parameters=vars(self.parameters), logging=vars(logconfig),
                                                   run_type=rm_enum.TRANSFER_LEARNING, version="2.0")

        model = reinvent_model.Model.load_from_file(self.parameters.input_model_path)
        logger = TransferLearningLogger(self.config)
        self.runner = TransferLearningRunner(model=model, config=self.parameters, logger=logger)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _check_if_models_exists(self):
        # Make sure the correct number of models are generated
        files = [f for f in os.listdir(self.workfolder) if os.path.isfile(os.path.join(self.workfolder, f))]
        self.assertEqual(len(files), self.num_epochs)

        self.assertTrue(os.path.isfile(self.output_file))
        nums = list(range(1, self.num_epochs))
        for n in nums:
            self.assertTrue(os.path.isfile(f"{self.output_file}.{n}"))

    def test_transfer_learning_no_validation(self):
        self.runner.run()
        self._check_if_models_exists()

    def test_transfer_learning_with_validation(self):
        self.parameters.validation_smiles_path = SMILES_SET_PATH
        self.runner.run()
        self._check_if_models_exists()