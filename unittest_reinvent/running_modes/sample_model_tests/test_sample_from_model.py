import os
import shutil
import unittest

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from running_modes.configurations.compound_sampling.sample_from_model_configuration import SampleFromModelConfiguration
from running_modes.sampling.sample_from_model import SampleFromModelRunner
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, PRIOR_PATH
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum


class TestSampleFromModel(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        rm_enums = RunningModeEnum()
        lm_enums = LoggingModeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "sample.smi")
        self.num_smiles = 100
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.config = SampleFromModelConfiguration(model_path=PRIOR_PATH,
                                                   output_smiles_path=self.output_file, num_smiles=self.num_smiles, batch_size=100,
                                                   with_likelihood=False)
        self.logging = SamplingLoggerConfiguration(recipient=lm_enums.LOCAL,
                                                   logging_path=f"{self.workfolder}/log", job_name="test_job")
        self.configurationenvelope = GeneralConfigurationEnvelope(parameters=vars(self.config), logging=vars(self.logging),
                                                                  run_type=rm_enums.SAMPLING, version="2.0")

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _tab_in_output_file(self) -> bool:
        return any("\t" in line for line in open(self.output_file))

    def _check_file_and_count_lines(self):
        self.assertTrue(os.path.isfile(self.output_file))
        num_lines = sum(1 for _ in open(self.output_file))
        self.assertEqual(num_lines, self.num_smiles)

    def test_sample_from_model_without_likelihood(self):
        runner = SampleFromModelRunner(self.configurationenvelope, self.config)
        runner.run()

        self._check_file_and_count_lines()
        self.assertFalse(self._tab_in_output_file())

    def test_sample_from_model_with_likelihood(self):
        self.config.with_likelihood = True
        runner = SampleFromModelRunner(self.configurationenvelope, self.config)
        runner.run()

        self._check_file_and_count_lines()
        # The likelihood is separated from the SMILES by a \t character
        self.assertTrue(self._tab_in_output_file())
