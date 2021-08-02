from running_modes.configurations.logging.base_log_config import BaseLoggerConfiguration


class ReinforcementLoggerConfiguration(BaseLoggerConfiguration):
    resultdir: str
    logging_frequency: int = 0
