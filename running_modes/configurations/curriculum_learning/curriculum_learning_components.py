from dataclasses import dataclass


@dataclass
class CurriculumLearningComponents:
    """This class holds the necessary configuration components to run CL"""
    curriculum_learning: dict
    scoring_function: dict
    diversity_filter: dict
    inception: dict
