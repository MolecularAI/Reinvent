from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ...configurations.logging.scoring_log_configuration import ScoringLoggerConfiguration
from .local_scoring_logger import LocalScoringLogger
from .remote_scoring_logger import RemoteScoringLogger
from ....utils.enums.logging_mode_enum import LoggingModeEnum


class ScoringLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope):
        logging_mode_enum = LoggingModeEnum()
        scoring_config = ScoringLoggerConfiguration(**configuration.logging)
        if scoring_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalScoringLogger(configuration)
        else:
            logger = RemoteScoringLogger(configuration)
        return logger
