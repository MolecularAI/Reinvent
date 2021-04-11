from reinvent_chemistry.conversions import Conversions
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.validation.logging.validation_logger import ValidationLogger

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import PredictivePropertyComponent


class ValidationRunner:
    def __init__(self, main_config: GeneralConfigurationEnvelope, parameters: ComponentParameters):
        self.parameters = parameters
        self.logger = ValidationLogger(main_config)
        self.chemistry = Conversions()

    def run(self):
        try:
            component = PredictivePropertyComponent(self.parameters)
            query_smiles = ['COc1cccc2cc(C(=O)NCCCCN3CCN(c4cccc5nccnc54)CC3)oc21']
            query_mols = [self.chemistry.smile_to_mol(smile) for smile in query_smiles]
            component.calculate_score(query_mols)
            self.logger.model_is_valid = True
            self.logger.log_message(message="Valid model")
        except Exception as e:
            self.logger.model_is_valid = False
            self.logger.log_message(message="Invalid model")