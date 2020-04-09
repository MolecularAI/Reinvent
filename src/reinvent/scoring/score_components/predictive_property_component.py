import pickle

from typing import List

from model_container import ModelContainer
from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary


class PredictivePropertyComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.activity_model = self._load_model(parameters)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self.activity_model.predict_from_mols(molecules, self.parameters.specific_parameters)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _load_model(self, parameters: ComponentParameters):
        try:
            model_type = self.parameters.specific_parameters[self.component_specific_parameters.SCIKIT]
            activity_model = self._load_scikit_model(parameters, model_type)
        except:
            raise Exception(f"The loaded file {parameters.model_path} isn't a valid scikit-learn model")
        return activity_model

    def _load_scikit_model(self, parameters: ComponentParameters, model_type):
        with open(parameters.model_path, "rb") as f:
            scikit_model = pickle.load(f)
            packaged_model = ModelContainer(scikit_model, model_type, parameters.specific_parameters)
        return packaged_model
