from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLoggerConfiguration
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.reinforcement_learning.logging.link_logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.reinforcement_learning.logging.link_logging.local_bond_link_reinforcement_logger import \
    LocalBondLinkReinforcementLogger
from running_modes.reinforcement_learning.logging.link_logging.remote_bond_link_reinforcement_logger import \
    RemoteLinkReinforcementLogger


class BondLinkReinforcementLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope, log_config: ReinforcementLoggerConfiguration) \
            -> BaseReinforcementLogger:
        logging_mode_enum = LoggingModeEnum()

        if log_config.recipient == logging_mode_enum.LOCAL:
            logger_instance = LocalBondLinkReinforcementLogger(configuration, log_config)
        else:
            logger_instance = RemoteLinkReinforcementLogger(configuration, log_config)

        return logger_instance