from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.constructors.create_model_mode_constructor import CreateModelModeConstructor
from running_modes.constructors.curriculum_learning_mode_constructor import CurriculumLearningModeConstructor
from running_modes.constructors.reinforcement_learning_mode_constructor import ReinforcementLearningModeConstructor
from running_modes.constructors.sampling_mode_constructor import SamplingModeConstructor
from running_modes.constructors.scoring_mode_constructor import ScoringModeConstructor
from running_modes.constructors.transfer_learning_mode_constructor import TransferLearningModeConstructor
from running_modes.constructors.validation_mode_constructor import ValidationModeConstructor
from running_modes.enums.running_mode_enum import RunningModeEnum


class RunningMode:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        running_mode_enum = RunningModeEnum()
        _configuration = configuration
        if configuration.run_type == running_mode_enum.REINFORCEMENT_LEARNING:
            return ReinforcementLearningModeConstructor(configuration)
        if configuration.run_type == running_mode_enum.CURRICULUM_LEARNING:
            return CurriculumLearningModeConstructor(configuration)
        if configuration.run_type == running_mode_enum.TRANSFER_LEARNING:
            return TransferLearningModeConstructor(configuration)
        if configuration.run_type == running_mode_enum.SCORING:
            return ScoringModeConstructor(configuration)
        if configuration.run_type == running_mode_enum.SAMPLING:
            return SamplingModeConstructor(configuration)
        if configuration.run_type == running_mode_enum.CREATE_MODEL:
            return CreateModelModeConstructor(configuration)
        if configuration.run_type == running_mode_enum.VALIDATION:
            return ValidationModeConstructor(configuration)
        # if configuration.run_type == running_mode_enum.AUTOMATED_CURRICULUM_LEARNING:
        #     return AutomatedCurriculumLearningModeConstructor(configuration)
        else:
            raise TypeError(f"Requested run type: '{configuration.run_type}' is not implemented.")