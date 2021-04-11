

class AdaptiveLearningRateEnum:
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"
    CONSTANT = "constant"

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    def __setattr__(self, key, value):
        raise ValueError("Do not assign value to a AdaptiveLearningRateEnum field.")