from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ...configurations.logging.transfer_learning_log_configuration import TransferLearningLoggerConfig
from .base_transfer_learning_logger import BaseTransferLearningLogger
from .local_transfer_learning_logger import LocalTransferLearningLogger
from .remote_transfer_learning_logger import RemoteTransferLearningLogger
from ....utils.enums.logging_mode_enum import LoggingModeEnum


class TransferLearningLogger:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseTransferLearningLogger:
        logging_mode_enum = LoggingModeEnum()
        tl_config = TransferLearningLoggerConfig(**configuration.logging)
        if tl_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalTransferLearningLogger(configuration)
        else:
            logger = RemoteTransferLearningLogger(configuration)
        return logger