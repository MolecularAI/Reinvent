from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from .base_validation_logger import BaseValidationLogger


class LocalValidationLogger(BaseValidationLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)

    def log_message(self, message: str):
        self._common_logger.info(message)

