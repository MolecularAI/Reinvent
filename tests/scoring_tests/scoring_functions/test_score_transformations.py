import unittest

from scoring.score_transformations import TransformationFactory
from utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from utils.enums.transformation_type_enum import TransformationTypeEnum


class Test_score_transformations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tt_enum = TransformationTypeEnum()
        cls.csp_enum = ComponentSpecificParametersEnum()
        cls.v_list = [12.086, 0.0015, 7.9, 123.264, 77.80, 4.0, 111.12]

    def setUp(self):
        self.factory = TransformationFactory()

    # ---------
    # note, that the case where "TRANSFORMATION" is set to "False" is not handled here,
    # as this functionality is part of the model container, rather than the transformation
    # factory; however, the expected result is the same as for "test_no_transformation"
    # ---------

    def test_no_transformation(self):
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.NO_TRANSFORMATION}
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)

        # the scores are not changed (transformation is set to "NO_TRANSFORMATION")
        self.assertListEqual(
            [12.086000442504883, 0.001500000013038516, 7.900000095367432, 123.26399993896484, 77.80000305175781, 4.0,
             111.12000274658203],
            transformed_scores.tolist())

    def test_right_step_transformation(self):
        # use standard parameters
        # ---------
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.LOW: 4,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.RIGHT_STEP}
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)

        # the scores are transformed to be 1.0 if >= 4 and 0.0 otherwise
        self.assertListEqual([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             transformed_scores.tolist())

        # use different parameters (value of the step function between the two and the remaining value)
        # ---------
        specific_parameters[self.csp_enum.LOW] = 25
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                             transformed_scores.tolist())

    def test_left_step_transformation(self):
        # use standard parameters
        # ---------
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.LOW: 4,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.LEFT_STEP}
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)

        # the scores are transformed to be 1.0 if <= 4 and 0.0 otherwise
        self.assertListEqual([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             transformed_scores.tolist())

        # use different parameters (value of the step function between the two and the remaining value)
        # ---------
        specific_parameters[self.csp_enum.LOW] = 25
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                             transformed_scores.tolist())

    def test_step_transformation(self):
        # use standard parameters
        # ---------
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.LOW: 4,
                               self.csp_enum.HIGH: 14,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.STEP}
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)

        # the scores are transformed to be 1.0 if >= 4 and 0.0 otherwise
        self.assertListEqual([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], transformed_scores.tolist())

    def test_sigmoid_transformation(self):
        # use standard parameters
        # ---------
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.LOW: 4,
                               self.csp_enum.HIGH: 9,
                               self.csp_enum.K: 0.25,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.SIGMOID}
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual(
            [0.9983919262886047, 0.0005629961378872395, 0.8336624503135681, 1.0, 1.0, 0.05324021354317665, 1.0],
            transformed_scores.tolist())

        # use higher values (higher values lower, lower values higher)
        # ---------
        specific_parameters[self.csp_enum.HIGH] = 45
        specific_parameters[self.csp_enum.LOW] = 9
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual(
            [0.08434110134840012, 0.013162842020392418, 0.045039791613817215, 0.9999998211860657, 0.9997034668922424,
             0.024656718596816063, 0.9999985694885254],
            transformed_scores.tolist())

    def test_reverse_sigmoid_transformation(self):
        # use standard parameters
        # ---------
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.LOW: 4,
                               self.csp_enum.HIGH: 9,
                               self.csp_enum.K: 0.25,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.REVERSE_SIGMOID}
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual(
            [0.0016080556670203805, 0.9994369745254517, 0.1663375347852707, 0.0, 2.2387210631548548e-36,
             0.9467597603797913, 0.0],
            transformed_scores.tolist())

        # use higher values (higher values higher, lower values lower)
        # ---------
        specific_parameters[self.csp_enum.HIGH] = 45
        specific_parameters[self.csp_enum.LOW] = 9

        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual(
            [0.9156588912010193, 0.9868371486663818, 0.9549602270126343, 2.0653797605518776e-07, 0.0002965469320770353,
             0.9753432869911194, 1.4399012115973164e-06],
            transformed_scores.tolist())

    def test_double_sigmoid_transformation(self):
        # use standard parameters
        # ---------
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.LOW: 4,
                               self.csp_enum.HIGH: 9,
                               self.csp_enum.COEF_DIV: 100,
                               self.csp_enum.COEF_SI: 150,
                               self.csp_enum.COEF_SE: 150,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.DOUBLE_SIGMOID}
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual(
            [2.3495775167248212e-05, 1.0051932122223661e-06, 0.9781016111373901, 0.0, 0.0, 0.4999999701976776, 0.0],
            transformed_scores.tolist())

        # use less constrained parameters for "middle values"
        # ---------
        specific_parameters[self.csp_enum.COEF_DIV] = 200
        specific_parameters[self.csp_enum.COEF_SI] = 100
        specific_parameters[self.csp_enum.COEF_SE] = 100
        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=self.v_list[:],
                                                parameters=specific_parameters)
        self.assertListEqual(
            [0.02775370515882969, 0.009886257350444794, 0.7690339088439941, 0.0, 0.0, 0.4968476891517639, 0.0],
            transformed_scores.tolist())

    def test_invalid_transformation(self):
        specific_parameters = {self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.TRANSFORMATION_TYPE: "NOT_IMPLEMENTED_TRANSFORMATION"}
        try:
            transform_function = self.factory.get_transformation_function(specific_parameters)
        except Exception as e:
            self.assertEqual(type(e).__name__, "KeyError")
        else:
            self.fail("""Expected exception of type "KeyError" because of invalid transformation selection.""")

    def test_custom_interpolation(self):
        specific_parameters = {self.csp_enum.TRUNCATE_RIGHT: True,
                               self.csp_enum.TRUNCATE_LEFT: True,
                               self.csp_enum.INTERPOLATION_MAP: [{"origin": 0.0, "destination": 0.0},
                                                                 {"origin": 1.0, "destination": 1.0}],
                               self.csp_enum.TRANSFORMATION: True,
                               self.csp_enum.TRANSFORMATION_TYPE: self.tt_enum.CUSTOM_INTERPOLATION}

        transform_function = self.factory.get_transformation_function(specific_parameters)
        transformed_scores = transform_function(predictions=[-1, 0., 0.3, 0.7, 15],
                                                parameters=specific_parameters)

        self.assertListEqual([0, 0, 0.3, 0.7, 1], transformed_scores.tolist())
