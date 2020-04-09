from running_modes.configurations import GeneralConfigurationEnvelope, BaseLoggerConfiguration
from running_modes.validation.logging.local_validation_logger import LocalValidationLogger
from running_modes.validation.logging.remote_validation_logger import RemoteValidationLogger
from utils.enums.logging_mode_enum import LoggingModeEnum


class ValidationLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope):
        logging_mode_enum = LoggingModeEnum()
        scoring_config = BaseLoggerConfiguration(**configuration.logging)
        if scoring_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalValidationLogger(configuration)
        else:
            logger = RemoteValidationLogger(configuration)
        return logger
