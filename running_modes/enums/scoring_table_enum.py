from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringTableEnum:

    AGENTS = "agents"
    SCORES = "scores"
    SCORING_FUNCTIONS = "scoring_functions"
    COMPONENT_NAMES = "component_names"
