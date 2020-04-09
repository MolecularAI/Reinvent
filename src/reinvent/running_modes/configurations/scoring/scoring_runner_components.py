from dataclasses import dataclass


@dataclass
class ScoringRunnerComponents:
    """This class holds the necessary configuration components to do a scoring run."""

    scoring: dict
    scoring_function: dict
