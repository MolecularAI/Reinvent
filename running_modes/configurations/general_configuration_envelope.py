from dataclasses import dataclass


@dataclass
class GeneralConfigurationEnvelope:
    parameters: dict
    logging: dict
    run_type: str
    version: str

