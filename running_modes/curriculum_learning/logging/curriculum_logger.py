from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.curriculum_log_configuration import CurriculumLoggerConfiguration
from running_modes.curriculum_learning.logging.local_curriculum_logger import LocalCurriculumLogger
from running_modes.curriculum_learning.logging.remote_curriculum_logger import RemoteCurriculumLogger
from running_modes.curriculum_learning.logging.base_curriculum_logger import BaseCurriculumLogger
from running_modes.enums.logging_mode_enum import LoggingModeEnum


class CurriculumLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseCurriculumLogger:
        logging_mode_enum = LoggingModeEnum()
        rl_config = CurriculumLoggerConfiguration(**configuration.logging)
        if rl_config.recipient == logging_mode_enum.LOCAL:
            logger_instance = LocalCurriculumLogger(configuration)
        else:
            logger_instance = RemoteCurriculumLogger(configuration)
        return logger_instance
