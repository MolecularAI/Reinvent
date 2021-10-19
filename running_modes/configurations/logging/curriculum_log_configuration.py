from running_modes.configurations.logging.base_log_config import BaseLoggerConfiguration


class CurriculumLoggerConfiguration(BaseLoggerConfiguration):
    result_folder: str
    logging_frequency: int = 0
