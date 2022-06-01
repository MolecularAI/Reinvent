from running_modes.configurations.logging.base_log_config import BaseLoggerConfiguration


class CurriculumLoggerConfiguration(BaseLoggerConfiguration):
    result_folder: str
    logging_frequency: int = 0
    # rows and columns define the shape of the output grid of molecule images in tensorboard.
    rows: int = 4
    columns: int = 4
