from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from running_modes.sampling.logging.base_sampling_logger import BaseSamplingLogger
from running_modes.sampling.logging.local_sampling_logger import LocalSamplingLogger
from running_modes.sampling.logging.remote_sampling_logger import RemoteSamplingLogger
from utils.enums.logging_mode_enum import LoggingModeEnum


class SamplingLogger:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseSamplingLogger:
        logging_mode_enum = LoggingModeEnum()
        sampling_config = SamplingLoggerConfiguration(**configuration.logging)
        if sampling_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalSamplingLogger(configuration)
        else:
            logger = RemoteSamplingLogger(configuration)
        return logger