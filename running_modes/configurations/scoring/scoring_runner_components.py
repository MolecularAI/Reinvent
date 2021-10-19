from dataclasses import dataclass

from reinvent_scoring.scoring import ScoringFunctionParameters

from running_modes.configurations.scoring.scoring_runner_configuration import ScoringRunnerConfiguration


@dataclass
class ScoringRunnerComponents:
    """This class holds the necessary configuration components to do a scoring run."""

    scoring: ScoringRunnerConfiguration
    scoring_function: ScoringFunctionParameters
