from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.reinforcement_log_configuration import ReinforcementLoggerConfiguration
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.reinforcement_learning.logging.link_logging.bond_link_reinforcement_logger import \
    BondLinkReinforcementLogger
from running_modes.reinforcement_learning.logging.local_reinforcement_logger import LocalReinforcementLogger
from running_modes.reinforcement_learning.logging.remote_reinforcement_logger import RemoteReinforcementLogger


class ReinforcementLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope, log_config: ReinforcementLoggerConfiguration) \
            -> BaseReinforcementLogger:
        logging_mode_enum = LoggingModeEnum()
        model_type_enum = ModelTypeEnum()

        if configuration.model_type == model_type_enum.DEFAULT:
            if log_config.recipient == logging_mode_enum.LOCAL:
                logger_instance = LocalReinforcementLogger(configuration, log_config)
            else:
                logger_instance = RemoteReinforcementLogger(configuration, log_config)
        else:
            logger_instance = BondLinkReinforcementLogger(configuration, log_config)

        return logger_instance
