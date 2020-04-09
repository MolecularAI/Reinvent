

class AdaptiveLearningRateEnum():
    _EXPONENTIAL = "exponential"
    _ADAPTIVE = "adaptive"
    _CONSTANT = "constant"


    @property
    def EXPONENTIAL(self):
        return self._EXPONENTIAL

    @EXPONENTIAL.setter
    def EXPONENTIAL(self, value):
        raise ValueError("Do not assign value to a AdaptiveLearningRateEnum field")

    @property
    def ADAPTIVE(self):
        return self._ADAPTIVE

    @ADAPTIVE.setter
    def ADAPTIVE(self, value):
        raise ValueError("Do not assign value to a AdaptiveLearningRateEnum field")

    @property
    def CONSTANT(self):
        return self._CONSTANT

    @CONSTANT.setter
    def CONSTANT(self, value):
        raise ValueError("Do not assign value to a AdaptiveLearningRateEnum field")
