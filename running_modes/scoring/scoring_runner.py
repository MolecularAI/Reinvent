from reinvent_chemistry.file_reader import FileReader
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction
from reinvent_scoring.scoring.score_summary import FinalSummary

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.scoring.scoring_runner_configuration import ScoringRunnerConfiguration
from running_modes.scoring.logging.scoring_logger import ScoringLogger


class ScoringRunner:
    def __init__(self, configuration: GeneralConfigurationEnvelope, config: ScoringRunnerConfiguration,
                 scoring_function: BaseScoringFunction):
        self._scoring_function = scoring_function
        self._config = config
        self._logger = ScoringLogger(configuration)
        self._reader = FileReader([], None)

    def run(self):
        input_smiles = list(self._reader.read_delimited_file(file_path=self._config.input, randomize=False, standardize=False))
        score_summary: FinalSummary = self._scoring_function.get_final_score(input_smiles)

        self._logger.log_results(score_summary=score_summary)
        self._logger.log_out_input_configuration()
