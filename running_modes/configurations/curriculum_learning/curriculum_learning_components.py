from dataclasses import dataclass

from running_modes.configurations.automated_curriculum_learning.base_configuration import BaseConfiguration


@dataclass
class CurriculumLearningComponents(BaseConfiguration):
    """This class holds the necessary configuration components to run CL"""
    curriculum_learning: dict
    scoring_function: dict
    diversity_filter: dict
    inception: dict
