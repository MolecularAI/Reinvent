from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.dto import CurriculumOutcomeDTO


class NoCurriculumStrategy(BaseCurriculumStrategy):

    def run(self) -> CurriculumOutcomeDTO:
        step_counter = 0
        self.disable_prior_gradients()
        outcome_dto = CurriculumOutcomeDTO(self._agent, step_counter, successful_curriculum=True)
        return outcome_dto

    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        pass

    def save_and_flush_memory(self, agent, memory_name: str):
        self._logger.save_merging_state(agent, self._diversity_filter, name=memory_name)
        self._diversity_filter = DiversityFilter(self._parameters.diversity_filter)
