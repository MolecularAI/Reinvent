import unittest
import os
import shutil
import utils as utils_general
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, RANDOM_PRIOR_PATH
from utils.enums.running_mode_enum import RunningModeEnum
from utils.enums.logging_mode_enum import LoggingModeEnum
from running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from running_modes.sampling.logging.local_sampling_logger import LocalSamplingLogger
from running_modes.sampling.sample_from_model import SampleFromModelRunner
from running_modes.configurations.compound_sampling.sample_from_model_configuration import SampleFromModelConfiguration


class Test_sample_logger_tbx(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        utils_general.set_default_device_cuda()
        lm_enums = LoggingModeEnum()
        rm_enums = RunningModeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "sample.smi")
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.config = SampleFromModelConfiguration(model_path=RANDOM_PRIOR_PATH,
                                                   output_smiles_path=self.output_file, num_smiles=100, batch_size=100,
                                                   with_likelihood=True)
        self.logging = SamplingLoggerConfiguration(sender="http://10.59.162.10:8081", recipient=lm_enums.LOCAL,
                                                   logging_path=f"{self.workfolder}/log", job_name="test_job")
        self.configurationenvelope = GeneralConfigurationEnvelope(parameters=vars(self.config), logging=vars(self.logging),
                                                                  run_type=rm_enums.SAMPLING, version="2.0")
        self.logger = LocalSamplingLogger(self.configurationenvelope)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_tbx_with_real_data(self):
        runner = SampleFromModelRunner(self.configurationenvelope, self.config)
        runner.run()
        self.assertEqual(os.path.isfile(self.output_file), True)
