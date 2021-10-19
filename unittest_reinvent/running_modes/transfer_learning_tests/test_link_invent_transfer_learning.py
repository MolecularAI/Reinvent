import shutil
import unittest
import os

from running_modes.configurations import TransferLearningLoggerConfig, GeneralConfigurationEnvelope
from running_modes.configurations.transfer_learning.link_invent_learning_rate_configuration import \
    LinkInventLearningRateConfiguration
from running_modes.configurations.transfer_learning.link_invent_transfer_learning_configuration import \
    LinkInventTransferLearningConfiguration
from running_modes.constructors.transfer_learning_mode_constructor import TransferLearningModeConstructor
from running_modes.utils import set_default_device_cuda
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum

from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, SMILES_SET_LINK_INVENT_PATH, LINK_INVENT_PRIOR_PATH
from unittest_reinvent.fixtures.utils import count_empty_files


class TestLinkInventTransferLearning(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        lm_enum = LoggingModeEnum()
        rm_enum = RunningModeEnum()
        mt_enum = ModelTypeEnum()

        self.workfolder = os.path.join(MAIN_TEST_PATH, mt_enum.LINK_INVENT + rm_enum.TRANSFER_LEARNING)
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.log_dir = os.path.join(self.workfolder, "test_log")
        log_config = TransferLearningLoggerConfig(logging_path=self.log_dir, recipient=lm_enum.LOCAL,
                                                  job_name="test_job")
        self.lr_config = LinkInventLearningRateConfiguration()
        self.parameters = LinkInventTransferLearningConfiguration(empty_model=LINK_INVENT_PRIOR_PATH,
                                                                  output_path=self.workfolder,
                                                                  input_smiles_path=SMILES_SET_LINK_INVENT_PATH,
                                                                  validation_smiles_path=None,
                                                                  num_epochs=2,
                                                                  sample_size=10,
                                                                  learning_rate=self.lr_config)
        self.general_config = GeneralConfigurationEnvelope(model_type=mt_enum.LINK_INVENT, logging=vars(log_config),
                                                           run_type=rm_enum.TRANSFER_LEARNING, version="3.0",
                                                           parameters=vars(self.parameters))
        self.runner = TransferLearningModeConstructor(self.general_config)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _model_saved_and_logs_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(self.workfolder, self.parameters.model_file_name)))
        self.assertTrue(os.path.isdir(self.log_dir))
        self.assertEqual(count_empty_files(self.log_dir), 0)

    def test_no_validation(self):
        self.parameters.validation_smiles_path = None
        self.runner.run()
        self._model_saved_and_logs_exist()

    def test_with_validation(self):
        self.parameters.validation_smiles_path = SMILES_SET_LINK_INVENT_PATH
        self.runner.run()
        self._model_saved_and_logs_exist()

