from reinvent_scoring import TransformationParametersEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums.descriptor_types_enum import DescriptorTypesEnum
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums.transformation_type_enum import TransformationTypeEnum

from unittest_reinvent.fixtures.paths import ACTIVITY_REGRESSION


def create_activity_component_regression():
    sf_enum = ScoringFunctionComponentNameEnum()
    specific_parameters = _specific_parameters_regression_predictive_model(ACTIVITY_REGRESSION)
    parameters = ComponentParameters(component_type=sf_enum.PREDICTIVE_PROPERTY,
                                     name="activity",
                                     weight=1.,
                                     specific_parameters=specific_parameters)
    return parameters


def create_predictive_property_component_regression():
    sf_enum = ScoringFunctionComponentNameEnum()
    specific_parameters = _specific_parameters_regression_predictive_model(ACTIVITY_REGRESSION)
    parameters = ComponentParameters(component_type=sf_enum.PREDICTIVE_PROPERTY,
                                     name="predictive_property",
                                     weight=1.,
                                     specific_parameters=specific_parameters)
    return parameters

def _specific_parameters_regression_predictive_model(path=None):
    csp_enum = ComponentSpecificParametersEnum()
    transf_type = TransformationTypeEnum()
    descriptor_types = DescriptorTypesEnum()
    transform_params = TransformationParametersEnum
    specific_parameters = {csp_enum.TRANSFORMATION: {transform_params.HIGH: 9,
                           transform_params.LOW: 4,
                           transform_params.K: 0.25,
                           transform_params.TRANSFORMATION_TYPE: transf_type.SIGMOID},
                           csp_enum.SCIKIT: "regression",
                           "model_path": path,
                           csp_enum.DESCRIPTOR_TYPE: descriptor_types.ECFP_COUNTS
                           }
    return specific_parameters
