from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.validation.logging.base_validation_logger import BaseValidationLogger


class LocalValidationLogger(BaseValidationLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)

    def log_message(self, message: str):
        self._common_logger.info(message)

