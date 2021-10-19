from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringRunnerEnum:

    SMILES = "smiles"
    TOTAL_SCORE = "total_score"
    VALID = "valid"
