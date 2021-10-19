from dataclasses import dataclass

from reinvent_scoring import ScoringFunctionParameters


@dataclass
class CurriculumObjective:
    scoring_function: ScoringFunctionParameters
    score_threshold: float = 0.4
