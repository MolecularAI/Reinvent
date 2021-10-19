from dataclasses import dataclass

from running_modes.enums.model_type_enum import ModelTypeEnum


@dataclass
class GeneralConfigurationEnvelope:
    parameters: dict
    logging: dict
    run_type: str
    version: str
    model_type: str = ModelTypeEnum().DEFAULT

