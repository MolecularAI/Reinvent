
class TransformationTypeEnum():
    __DOUBLE_SIGMOID = "double_sigmoid"
    __SIGMOID = "sigmoid"
    __REVERSE_SIGMOID = "reverse_sigmoid"
    __RIGHT_STEP = "right_step"
    __LEFT_STEP = "left_step"
    __STEP = "step"
    __CUSTOM_INTERPOLATION = "custom_interpolation"
    __NO_TRANSFORMATION = "no_transformation"

    @property
    def DOUBLE_SIGMOID(self):
        return self.__DOUBLE_SIGMOID

    @DOUBLE_SIGMOID.setter
    def DOUBLE_SIGMOID(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def SIGMOID(self):
        return self.__SIGMOID

    @SIGMOID.setter
    def SIGMOID(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def REVERSE_SIGMOID(self):
        return self.__REVERSE_SIGMOID

    @REVERSE_SIGMOID.setter
    def REVERSE_SIGMOID(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def RIGHT_STEP(self):
        return self.__RIGHT_STEP

    @RIGHT_STEP.setter
    def RIGHT_STEP(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def LEFT_STEP(self):
        return self.__LEFT_STEP

    @LEFT_STEP.setter
    def LEFT_STEP(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def STEP(self):
        return self.__STEP

    @STEP.setter
    def STEP(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def CUSTOM_INTERPOLATION(self):
        return self.__CUSTOM_INTERPOLATION

    @CUSTOM_INTERPOLATION.setter
    def CUSTOM_INTERPOLATION(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def NO_TRANSFORMATION(self):
        return self.__NO_TRANSFORMATION

    @NO_TRANSFORMATION.setter
    def NO_TRANSFORMATION(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")