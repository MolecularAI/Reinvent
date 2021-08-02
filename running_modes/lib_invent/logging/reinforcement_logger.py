from running_modes.lib_invent.configurations.log_configuration import LogConfiguration
from running_modes.lib_invent.logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.lib_invent.logging.local_reinforcement_logger import LocalReinforcementLogger
from running_modes.enums.logging_mode_enum import LoggingModeEnum


class ReinforcementLogger:

    def __new__(cls, configuration: LogConfiguration) -> BaseReinforcementLogger:
        logging_mode_enum = LoggingModeEnum()
        if configuration.recipient == logging_mode_enum.LOCAL:
            return LocalReinforcementLogger(configuration)
        else:
            raise NotImplemented("Remote logging mode is not implemented yet !")
