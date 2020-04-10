import json
import os

import pandas as pd

from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from .base_scoring_logger import BaseScoringLogger
from ....scoring.score_summary import FinalSummary
from ....utils.enums.scoring_runner_enum import ScoringRunnerEnum


class LocalScoringLogger(BaseScoringLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._scoring_runner_enum = ScoringRunnerEnum()

    def log_message(self, message: str):
        self._logger.info(message)

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def log_results(self, score_summary: FinalSummary):
        output_file = os.path.join(self._log_config.logging_path, "scored_smiles.csv")
        component_names = [c.name for c in score_summary.profile]

        data_list = self._convolute_score_summary(score_summary)
        data_df = self._construct_df_from_list(data=data_list, component_names=component_names)
        data_df.to_csv(path_or_buf=output_file, sep=',', header=True, index=False)

    @staticmethod
    def _convolute_score_summary(score_summary: FinalSummary) -> []:
        #TODO: seems like this can benefit from some refactoring
        """iterate over all smiles and extract scores, components and validity for each"""
        smiles = score_summary.scored_smiles
        component_scores = [c.score for c in score_summary.profile]
        data = []

        for i_smile in range(len(smiles)):
            score = '0'
            valid = '0'

            if i_smile in score_summary.valid_idxs:
                score = str(score_summary.total_score[i_smile])
                valid = '1'

            row = [smiles[i_smile], score]
            for component in component_scores:
                row.append(component[i_smile])
            row.append(valid)

            data.append(row)

        return data

    def _construct_df_from_list(self, data: list, component_names: list) -> pd.DataFrame:

        column_names = [self._scoring_runner_enum.SMILES, self._scoring_runner_enum.TOTAL_SCORE]
        column_names.extend(component_names)
        column_names.append(self._scoring_runner_enum.VALID)

        dataframe = pd.DataFrame(data, columns=column_names, dtype=str)

        return dataframe