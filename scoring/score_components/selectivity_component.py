import pickle

from typing import List

from model_container import ModelContainer
from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary
from scoring.score_transformations import TransformationFactory


class SelectivityComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._activity_params = self._prepare_activity_parameters(parameters)
        self._off_target_params = self._prepare_offtarget_parameters(parameters)
        self._activity_model = self._load_model(self._activity_params)
        self._off_target_activity_model = self._load_model(self._off_target_params)
        self._delta_params = self._prepare_delta_parameters(parameters)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score, offtarget_score = self._calculate_offtarget_activity(molecules, self._activity_params,
                                                                    self._off_target_params, self._delta_params)
        score_summary = ComponentSummary(total_score=score, parameters=self._off_target_params)
        return score_summary

    def _load_model(self, parameters: ComponentParameters):
        try:
            model_type = parameters.specific_parameters[self.component_specific_parameters.SCIKIT]
            activity_model = self._load_scikit_model(parameters, model_type)
        except:
            raise Exception(f"The loaded file {parameters.model_path} isn't a valid scikit-learn model")
        return activity_model

    def _load_scikit_model(self, parameters: ComponentParameters, model_type) -> ModelContainer:
        with open(parameters.model_path, "rb") as f:
            scikit_model = pickle.load(f)

            models_are_identical = self._activity_params.specific_parameters[
                                       self.component_specific_parameters.SCIKIT] == \
                                   self._off_target_params.specific_parameters[
                                       self.component_specific_parameters.SCIKIT]

            model_is_regression = self._off_target_params.specific_parameters[
                                      self.component_specific_parameters.SCIKIT] == "regression"

            both_models_are_regression = models_are_identical and model_is_regression

            if both_models_are_regression:
                parameters.specific_parameters[self.component_specific_parameters.TRANSFORMATION] = False
            packaged_model = ModelContainer(scikit_model, model_type, parameters.specific_parameters)
        return packaged_model

    def _calculate_offtarget_activity(self, molecules, activity_params, offtarget_params, delta_params):
        activity_score = self._activity_model.predict_from_mols(molecules, activity_params.specific_parameters)
        offtarget_score = self._off_target_activity_model.predict_from_mols(molecules,
                                                                            offtarget_params.specific_parameters)
        delta = activity_score - offtarget_score

        t_function = self._assign_transformation(delta_params)
        transformed_score = t_function(delta, delta_params) if delta_params[
            self.component_specific_parameters.TRANSFORMATION] else delta
        transformed_score[transformed_score < 0.01] = 0.01

        return transformed_score, offtarget_score

    def _assign_transformation(self, specific_parameters: {}):
        factory = TransformationFactory()
        transform_function = factory.get_transformation_function(specific_parameters)
        return transform_function

    def _prepare_activity_parameters(self, parameters: ComponentParameters) -> ComponentParameters:
        model_path = parameters.specific_parameters["activity_model_path"]
        specific_params = parameters.specific_parameters["activity_specific_parameters"]
        activity_params = ComponentParameters(name=self.parameters.name,
                                              weight=self.parameters.weight,
                                              smiles=self.parameters.smiles,
                                              model_path=model_path,
                                              component_type=self.parameters.component_type,
                                              specific_parameters=specific_params
                                              )
        return activity_params

    def _prepare_offtarget_parameters(self, parameters: ComponentParameters) -> ComponentParameters:
        model_path = parameters.specific_parameters["offtarget_model_path"]
        specific_params = parameters.specific_parameters["offtarget_specific_parameters"]
        offtarget_params = ComponentParameters(name=self.parameters.name,
                                               weight=self.parameters.weight,
                                               smiles=self.parameters.smiles,
                                               model_path=model_path,
                                               component_type=self.parameters.component_type,
                                               specific_parameters=specific_params
                                               )
        return offtarget_params

    def _prepare_delta_parameters(self, parameters: ComponentParameters) -> dict:
        specific_params = parameters.specific_parameters["delta_transformation_parameters"]
        specific_params[self.component_specific_parameters.TRANSFORMATION] = \
            "regression" == self._activity_params.specific_parameters[self.component_specific_parameters.SCIKIT] == \
            self._off_target_params.specific_parameters[self.component_specific_parameters.SCIKIT]
        return specific_params
