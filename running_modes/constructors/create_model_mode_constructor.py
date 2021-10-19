from dacite import from_dict

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope, CreateModelConfiguration, \
    LinkInventCreateModelConfiguration
from running_modes.create_model import CreateModelRunner, LinkInventCreateModelRunner
from running_modes.create_model.logging.create_model_logger import CreateModelLogger
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.utils.general import set_default_device_cuda


class CreateModelModeConstructor:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        cls._configuration = configuration
        set_default_device_cuda()
        logger = CreateModelLogger(cls._configuration)
        model_type_enum = ModelTypeEnum()
        if cls._configuration.model_type == model_type_enum.DEFAULT:
            config = from_dict(data_class=CreateModelConfiguration, data=cls._configuration.parameters)
            runner = CreateModelRunner(configuration=config, logger=logger)
        elif cls._configuration.model_type == model_type_enum.LIB_INVENT:
            raise NotImplementedError(f"Running mode not implemented for a model type: {cls._configuration.model_type}")
        elif cls._configuration.model_type == model_type_enum.LINK_INVENT:
            config = from_dict(data_class=LinkInventCreateModelConfiguration, data=cls._configuration.parameters)
            runner = LinkInventCreateModelRunner(configuration=config, logger=logger)
        else:
            raise ValueError(f"Incorrect model type: `{cls._configuration.model_type}` provided")

        return runner
