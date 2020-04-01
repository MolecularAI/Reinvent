from scoring.component_parameters import ComponentParameters
from scoring.function import CustomProduct, CustomSum
from scoring.function.base_scoring_function import BaseScoringFunction
from scoring.scoring_function_parameters import ScoringFuncionParameters
from utils.enums.scoring_function_enum import ScoringFunctionNameEnum


class ScoringFunctionFactory:

    def __new__(cls, sf_parameters: ScoringFuncionParameters) -> BaseScoringFunction:
        enum = ScoringFunctionNameEnum()
        scoring_function_registry = {
            enum.CUSTOM_PRODUCT: CustomProduct,
            enum.CUSTOM_SUM: CustomSum
        }
        return cls.create_scoring_function_instance(sf_parameters, scoring_function_registry)

    @staticmethod
    def create_scoring_function_instance(sf_parameters: ScoringFuncionParameters,
                                         scoring_function_registry: dict) -> BaseScoringFunction:
        """Returns a scoring function instance"""
        scoring_function = scoring_function_registry[sf_parameters.name]
        parameters = [ComponentParameters(**p) for p in sf_parameters.parameters]

        return scoring_function(parameters, sf_parameters.parallel)
