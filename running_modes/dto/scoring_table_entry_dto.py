from dataclasses import dataclass
from typing import Any


@dataclass
class ScoringTableEntryDTO:
    agent: Any
    score: float
    scoring_function_components: Any