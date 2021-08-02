import os
import shutil
import unittest


import running_modes.utils.general as utils_general
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from running_modes.configurations.compound_sampling.sample_from_model_configuration import SampleFromModelConfiguration
from running_modes.sampling.logging.local_sampling_logger import LocalSamplingLogger
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, PRIOR_PATH
from unittest_reinvent.fixtures.utils import count_empty_files
from unittest_reinvent.fixtures.test_data import ASPIRIN, PROPANE, IBUPROFEN, CAFFEINE, GENTAMICIN, REP_LIKELIHOOD, \
    REP_SMILES_LIST, LIKELIHOODLIST, INVALID_SMILES_LIST
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum

class TestSamplingInvalidSmiles(unittest.TestCase):

    def setUp(self):
        lm_enums = LoggingModeEnum()
        utils_general.set_default_device_cuda()
        rm_enums = RunningModeEnum()
        self.smiles = [ASPIRIN, PROPANE, IBUPROFEN, CAFFEINE, GENTAMICIN]
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "sample.smi")
        self.logging_path = f"{self.workfolder}/SPlog"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.config = SampleFromModelConfiguration(model_path=PRIOR_PATH,
                                                   output_smiles_path=self.output_file, num_smiles=100, batch_size=100,
                                                   with_likelihood=True)
        self.logging = SamplingLoggerConfiguration(recipient=lm_enums.LOCAL,
                                                   logging_path=self.logging_path,
                                                   job_name="test_job")
        configuration = GeneralConfigurationEnvelope(parameters=vars(self.config), logging=vars(self.logging),
                                                     run_type=rm_enums.SAMPLING, version="2.0")
        self.logger = LocalSamplingLogger(configuration)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _check_log_directory(self):
        self.assertEqual(os.path.isdir(self.logging_path), True)
        self.assertEqual(len(os.listdir(self.logging_path)) >= 1, True)
        self.assertEqual(count_empty_files(self.logging_path), 0)

    def test_sampling_logging_with_only_invalids(self):
        self.logger.timestep_report(INVALID_SMILES_LIST, REP_LIKELIHOOD)
        self._check_log_directory()

    def test_sampling_logging_with_some_valids(self):
        self.logger.timestep_report(self.smiles, LIKELIHOODLIST[1:5])
        self._check_log_directory()

    def test_sampling_logging_with_duplicate_valids(self):
        self.logger.timestep_report(REP_SMILES_LIST, REP_LIKELIHOOD)
        self._check_log_directory()
