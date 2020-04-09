from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ...configurations.logging.reinforcement_log_configuration import ReinforcementLoggerConfiguration
from .base_reinforcement_logger import BaseReinforcementLogger
from .local_reinforcement_logger import LocalReinforcementLogger
from .remote_reinforcement_logger import RemoteReinforcementLogger
from ....utils.enums.logging_mode_enum import LoggingModeEnum


class ReinforcementLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseReinforcementLogger:
        logging_mode_enum = LoggingModeEnum()
        rl_config = ReinforcementLoggerConfiguration(**configuration.logging)
        if rl_config.recipient == logging_mode_enum.LOCAL:
            logger_instance = LocalReinforcementLogger(configuration)
        else:
            logger_instance = RemoteReinforcementLogger(configuration)
        return logger_instance
