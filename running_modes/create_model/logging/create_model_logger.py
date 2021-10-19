from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLoggerConfiguration
from running_modes.create_model.logging.base_create_model_logger import BaseCreateModelLogger
from running_modes.create_model.logging.local_create_model_logger import LocalCreateModelLogger
from running_modes.create_model.logging.remote_create_model_logger import RemoteCreateModelLogger
from running_modes.enums.logging_mode_enum import LoggingModeEnum


class CreateModelLogger:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseCreateModelLogger:
        logging_mode_enum = LoggingModeEnum()
        logging_config = ReinforcementLoggerConfiguration.parse_obj(configuration.logging)
        if logging_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalCreateModelLogger(configuration)
        elif logging_config.recipient == logging_mode_enum.REMOTE:
            logger = RemoteCreateModelLogger(configuration)
        else:
            raise ValueError(f"Incorrect logger recipient: `{logging_config.recipient}` provided")
        return logger
