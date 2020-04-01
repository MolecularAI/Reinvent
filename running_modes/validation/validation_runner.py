from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.validation.logging.validation_logger import ValidationLogger
from scoring.component_parameters import ComponentParameters
from scoring.score_components import PredictivePropertyComponent

import utils.smiles as chem_smiles


class ValidationRunner:
    def __init__(self, main_config: GeneralConfigurationEnvelope, parameters: ComponentParameters):
        self.parameters = parameters
        self.logger = ValidationLogger(main_config)

    def run(self):
        try:
            component = PredictivePropertyComponent(self.parameters)
            query_smiles = ['O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N']
            query_mols = [chem_smiles.to_mol(smile) for smile in query_smiles]
            component.calculate_score(query_mols)
            self.logger.model_is_valid = True
            self.logger.log_message(message="Valid model")
        except Exception as e:
            self.logger.model_is_valid = False
            self.logger.log_message(message="Invalid model")