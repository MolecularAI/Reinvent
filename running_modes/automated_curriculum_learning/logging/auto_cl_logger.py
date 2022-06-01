from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum

from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.automated_curriculum_learning.logging.local_logger import LocalLogger
from running_modes.configurations import CurriculumLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.enums.logging_mode_enum import LoggingModeEnum


class AutoCLLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseLogger:
        logging_mode_enum = LoggingModeEnum()
        model_type = ModelTypeEnum()
        auto_cl_config = CurriculumLoggerConfiguration.parse_obj(configuration.logging)
        
        if auto_cl_config.recipient == logging_mode_enum.LOCAL:
            if model_type.DEFAULT == configuration.model_type:
                return LocalLogger(configuration, auto_cl_config)
            elif model_type.LINK_INVENT == configuration.model_type:
                return LocalLogger(configuration, auto_cl_config)
        else:
            raise NotImplementedError("Remote Auto CL logging is not implemented.")

