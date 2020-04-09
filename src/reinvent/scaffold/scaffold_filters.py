import abc
import os

import pandas as pd
from typing import List

from .scaffold_parameters import ScaffoldParameters
from ..scoring.score_summary import FinalSummary, ComponentSummary
from ..utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class ScaffoldFilter(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, parameters: ScaffoldParameters):
        self._scaffolds = {}
        self.parameters = parameters

    @property
    def scaffolds(self):
        return self._scaffolds

    @scaffolds.setter
    def scaffolds(self, value):
        raise AttributeError("""Parameter "scaffolds" is read-only.""")

    def _calculate_scaffold(self, smile):
        raise NotImplementedError

    @abc.abstractmethod
    def score(self, score_summary: FinalSummary):
        pass

    def _smiles_exists(self, scaffold, smile):
        # TODO: the if statement below will seal the scaffold bucket.
        # if scaffold in self.scaffolds and len(self.scaffolds[scaffold]) <= self.nbmax:
        # Otherwise the current statement enables to add scaffolds while still penalizing them.
        if scaffold in self.scaffolds:
            if smile in self.scaffolds[scaffold]:
                return True
        return False

    def _add_to_memory(self, indx: int, score, smile, cluster, components: List[ComponentSummary]):
        sf_enum = ScoringFunctionComponentNameEnum()
        component_scores = {c.parameters.name: float(c.total_score[indx]) for c in components}
        component_scores[sf_enum.TOTAL_SCORE] = float(score)
        self._update_memory(smile, cluster, component_scores)

    def _update_memory(self, smile, scaffold, score=None):
        if scaffold in self.scaffolds:
            self.scaffolds[scaffold][smile] = score
        else:
            self.scaffolds[scaffold] = {smile: score}

    def save_to_csv(self, path, job_name="default_job"):
        # TODO: scaffold should be passed as a paremeter
        # TODO: consider moving this method to a logger class
        sf_enum = ScoringFunctionComponentNameEnum()
        if not os.path.isdir(path):
            os.makedirs(path)
        df_dict = {"Cluster": [], "Scaffold": [], "SMILES": []}
        for i, scaffold in enumerate(self._scaffolds):
            for smi, score in self._scaffolds[scaffold].items():
                df_dict["Cluster"].append(i)
                df_dict["Scaffold"].append(scaffold)
                df_dict["SMILES"].append(smi)
                for item in score.keys():
                    if item in df_dict:
                        df_dict[item].append(score[item])
                    else:
                        df_dict[item] = [score[item]]
        file = os.path.join(path, "scaffold_memory.csv")
        raw_df = pd.DataFrame(df_dict)
        if len(raw_df) > 0:
            sorted_df = raw_df.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
            sorted_df["ID"] = [f"{job_name}_{e}" for e, _ in enumerate(sorted_df.index.array)]
            sorted_df["job_name"] = [f"{job_name}" for _ in enumerate(sorted_df.index.array)]
            sorted_df.to_csv(file, index=False)

    def _penalize_score(self, cluster, score):
        def _bucket_size(scaffold):
            # TODO: this should be probably a try catch block
            if scaffold in self.scaffolds:
                return len(self.scaffolds[scaffold])
            else:
                return 0
        # penalizes the score if the scaffold bucket is full
        if _bucket_size(cluster) > self.parameters.nbmax:
            score = 0
        return score













