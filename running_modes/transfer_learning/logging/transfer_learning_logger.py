from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.transfer_learning_log_configuration import TransferLearningLoggerConfig
from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger
from running_modes.transfer_learning.logging.local_link_invent_transfer_learning_logger import \
    LocalLinkInventTransferLearningLogger
from running_modes.transfer_learning.logging.local_transfer_learning_logger import LocalTransferLearningLogger
from running_modes.transfer_learning.logging.remote_transfer_learning_logger import RemoteTransferLearningLogger
from running_modes.enums.logging_mode_enum import LoggingModeEnum


class TransferLearningLogger:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseTransferLearningLogger:
        logging_mode_enum = LoggingModeEnum()
        model_type_enum = ModelTypeEnum()

        tl_config = TransferLearningLoggerConfig(**configuration.logging)
        if tl_config.recipient == logging_mode_enum.LOCAL and configuration.model_type == model_type_enum.LINK_INVENT:
            logger = LocalLinkInventTransferLearningLogger(configuration)
        elif tl_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalTransferLearningLogger(configuration)
        else:
            logger = RemoteTransferLearningLogger(configuration)
        # else:
        #     raise ValueError(f'Invalid logger type : {tl_config.recipient}')
        return logger
