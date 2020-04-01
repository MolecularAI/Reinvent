class ROCSSpecificParametersEnum():
    _ROCS_INPUT = "rocs_input"
    _INPUT_TYPE = "input_type"
    _SHAPE_WEIGHT = "shape_weight"
    _COLOR_WEIGHT = "color_weight"
    _SIM_MEASURE = "similarity_measure"
    _MAX_CPUS = "max_num_cpus"

    # Need to stick to this pattern of getter/setter as the alternative more concise one cannot
    # be pickled for multiprocessing

    @property
    def ROCS_INPUT(self):
        return self._ROCS_INPUT

    @ROCS_INPUT.setter
    def ROCS_INPUT(self, value):
        raise ValueError("Do not assign value to a ROCSSpecificParametersEnum field")

    @property
    def INPUT_TYPE(self):
        return self._INPUT_TYPE

    @INPUT_TYPE.setter
    def INPUT_TYPE(self, value):
        raise ValueError("Do not assign value to a ROCSSpecificParametersEnum field")

    @property
    def SHAPE_WEIGHT(self):
        return self._SHAPE_WEIGHT

    @SHAPE_WEIGHT.setter
    def SHAPE_WEIGHT(self, value):
        raise ValueError("Do not assign value to a ROCSSpecificParametersEnum field")

    @property
    def COLOR_WEIGHT(self):
        return self._COLOR_WEIGHT

    @COLOR_WEIGHT.setter
    def COLOR_WEIGHT(self, value):
        raise ValueError("Do not assign value to a ROCSSpecificParametersEnum field")

    @property
    def SIM_MEASURE(self):
        return self._SIM_MEASURE

    @SIM_MEASURE.setter
    def SIM_MEASURE(self, value):
        raise ValueError("Do not assign value to a ROCSSpecificParametersEnum field")

    @property
    def MAX_CPUS(self):
        return self._MAX_CPUS

    @MAX_CPUS.setter
    def MAX_CPUS(self, value):
        raise ValueError("Do not assign value to a ROCSSpecificParametersEnum field")
