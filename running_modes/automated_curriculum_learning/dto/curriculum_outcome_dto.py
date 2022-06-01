from dataclasses import dataclass
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase


@dataclass
class CurriculumOutcomeDTO:
    agent: GenerativeModelBase
    step_counter: int
    successful_curriculum: bool
