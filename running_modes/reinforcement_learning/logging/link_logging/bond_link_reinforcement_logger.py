from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLoggerConfiguration
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.reinforcement_learning.logging.link_logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.reinforcement_learning.logging.link_logging.local_bond_link_reinforcement_logger import \
    LocalBondLinkReinforcementLogger


class BondLinkReinforcementLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope, log_config: ReinforcementLoggerConfiguration) \
            -> BaseReinforcementLogger:
        logging_mode_enum = LoggingModeEnum()

        if log_config.recipient == logging_mode_enum.LOCAL:
            logger_instance = LocalBondLinkReinforcementLogger(configuration, log_config)
        else:
            raise NotImplemented("Remote logging mode is not implemented yet !")

        return logger_instance