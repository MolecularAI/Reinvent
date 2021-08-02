from running_modes.configurations.logging.base_log_config import BaseLoggerConfiguration


class CurriculumLoggerConfiguration(BaseLoggerConfiguration):
    resultdir: str
    logging_frequency: int = 0
