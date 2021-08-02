import unittest
from typing import List

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.validation.validation_runner import ValidationRunner
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, ACTIVITY_REGRESSION
from running_modes.configurations.logging.base_log_config import BaseLoggerConfiguration

from unittest_reinvent.fixtures.predictive_model_fixtures import create_activity_component_regression

from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums.transformation_type_enum import TransformationTypeEnum
from reinvent_scoring.scoring.enums.descriptor_types_enum import DescriptorTypesEnum

from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.enums.logging_mode_enum import LoggingModeEnum


class TestModelValidity(unittest.TestCase):
    """
    Unit test to ascertain that the command line check for model validity (distinct running mode) works as expected
    """
    def setUp(self):
        self.rm_enums = RunningModeEnum()
        self.cs_enum = ComponentSpecificParametersEnum()
        self.tf_enum = TransformationTypeEnum()
        self.sfc_enum = ScoringFunctionComponentNameEnum()
        self.lm_enum = LoggingModeEnum()
        self.parameters = create_activity_component_regression()
        log_conf = BaseLoggerConfiguration(recipient=self.lm_enum.LOCAL,
                                           logging_path=f"{MAIN_TEST_PATH}/log",
                                           job_name="model_validation_test",
                                           job_id="1")
        self.configuration_envelope = GeneralConfigurationEnvelope(parameters=vars(self.parameters),
                                                                   logging=vars(log_conf),
                                                                   run_type=self.rm_enums.VALIDATION,
                                                                   version="2.0")

    def _assert_output(self, expected: List[str]):
        runner = ValidationRunner(self.configuration_envelope, self.parameters)

        with self.assertLogs() as cm:
            runner.run()
        self.assertEqual(cm.output, expected)

    def test_valid_model(self):
        self._assert_output(["INFO:validation_logger:Valid model"])

    def test_invalid_model(self):
        self.parameters.model_path = "".join([ACTIVITY_REGRESSION, "_NO-MODEL_"])
        self._assert_output(["INFO:validation_logger:Invalid model"])

    def test_invalid_model_type(self):
        self.parameters.specific_parameters[self.cs_enum.SCIKIT] = "classification"
        self._assert_output(["INFO:validation_logger:Invalid model"])

    def test_invalid_model_descriptors(self):
        descriptor_types = DescriptorTypesEnum()
        self.parameters.specific_parameters[self.cs_enum.DESCRIPTOR_TYPE] = descriptor_types.ECFP
        self.parameters.specific_parameters["size"] = 1024
        self._assert_output(["INFO:validation_logger:Invalid model"])
