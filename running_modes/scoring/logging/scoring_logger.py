from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.scoring_log_configuration import ScoringLoggerConfiguration
from running_modes.scoring.logging.local_scoring_logger import LocalScoringLogger
from running_modes.scoring.logging.remote_scoring_logger import RemoteScoringLogger
from running_modes.enums.logging_mode_enum import LoggingModeEnum


class ScoringLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope):
        logging_mode_enum = LoggingModeEnum()
        scoring_config = ScoringLoggerConfiguration(**configuration.logging)
        if scoring_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalScoringLogger(configuration)
        else:
            logger = RemoteScoringLogger(configuration)
        return logger
