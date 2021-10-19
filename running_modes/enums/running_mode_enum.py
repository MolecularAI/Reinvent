from dataclasses import dataclass


@dataclass(frozen=True)
class RunningModeEnum:
    TRANSFER_LEARNING = "transfer_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SAMPLING = "sampling"
    CREATE_MODEL = "create_model"
    VALIDATION = "validation"
    SCORING = "scoring"
    CURRICULUM_LEARNING = "curriculum_learning"
    LIB_INVENT_REINFORCEMENT_LEARNING = "lib_invent_reinforcement_learning"
