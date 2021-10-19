from running_modes.automated_curriculum_learning.logging import LocalAutoCLLogger
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.configurations import ReinforcementLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope

from running_modes.enums.logging_mode_enum import LoggingModeEnum


class AutoCLLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseAutoCLLogger:
        logging_mode_enum = LoggingModeEnum()
        auto_cl_config = ReinforcementLoggerConfiguration.parse_obj(configuration.logging)
        
        if auto_cl_config.recipient == logging_mode_enum.LOCAL:
            logger_instance = LocalAutoCLLogger(configuration)

        else:
            raise NotImplementedError("Remote Auto CL logging is not implemented.")
        return logger_instance
