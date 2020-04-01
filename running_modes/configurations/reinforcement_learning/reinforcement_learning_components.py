from dataclasses import dataclass


@dataclass
class ReinforcementLearningComponents:
    """This class holds the necessary configuration components to run RL"""
    reinforcement_learning: dict
    scoring_function: dict
    scaffold_filter: dict
    inception: dict
