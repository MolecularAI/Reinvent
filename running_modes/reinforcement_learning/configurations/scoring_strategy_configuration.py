from dataclasses import dataclass

from reinvent_scoring import ScoringFunctionParameters


@dataclass
class ScoringStrategyConfiguration:
    scoring_function: ScoringFunctionParameters
    name: str
