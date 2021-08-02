
class RunningModeEnum:
    TRANSFER_LEARNING = "transfer_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SAMPLING = "sampling"
    CREATE_MODEL = "create_model"
    VALIDATION = "validation"
    SCORING = "scoring"
    CURRICULUM_LEARNING = "curriculum_learning"
    LIB_INVENT_REINFORCEMENT_LEARNING = "lib_invent_reinforcement_learning"
    AUTOMATED_CURRICULUM_LEARNING = "automated_curriculum_learning"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
