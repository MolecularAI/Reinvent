import time
from typing import List, Tuple

import numpy as np
import torch
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.actions.reinvent_sample_model import ReinventSampleModel
from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.dto import SampledBatchDTO, CurriculumOutcomeDTO, TimestepDTO


class ReinventCurriculumStrategy(BaseCurriculumStrategy):

    def run(self) -> CurriculumOutcomeDTO:
        step_counter = 0
        self.disable_prior_gradients()

        for item_id, sf_configuration in enumerate(self._parameters.curriculum_objectives):
            start_time = time.time()
            scoring_function = self._setup_scoring_function(item_id)
            step_counter = self.promote_agent(agent=self._agent, scoring_function=scoring_function,
                                              step_counter=step_counter, start_time=start_time,
                                              merging_threshold=sf_configuration.score_threshold)
            self.save_and_flush_memory(agent=self._agent, memory_name=f"_merge_{item_id}")
        is_successful_curriculum = step_counter < self._parameters.max_num_iterations
        outcome_dto = CurriculumOutcomeDTO(self._agent, step_counter, successful_curriculum=is_successful_curriculum)

        return outcome_dto

    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        # 1. Sampling
        sampled = self._sampling(agent)
        # 2. Scoring
        score, score_summary = self._scoring(scoring_function, sampled.smiles, step)
        # 3. Updating
        agent_likelihood, prior_likelihood, augmented_likelihood = self._updating(sampled, score, self.inception, agent)
        # 4. Logging
        self._logging(agent=agent, start_time=start_time, step=step,
                      score_summary=score_summary, agent_likelihood=agent_likelihood,
                      prior_likelihood=prior_likelihood, augmented_likelihood=augmented_likelihood)

        score = score.mean()
        return score

    def _sampling(self, agent) -> SampledBatchDTO:
        sampling_action = ReinventSampleModel(agent, self._parameters.batch_size, self._logger)
        sampled_sequences = sampling_action.run()
        return sampled_sequences

    def _scoring(self, scoring_function, smiles: List[str], step) -> Tuple[np.ndarray, FinalSummary] :
        score_summary = scoring_function.get_final_score_for_step(smiles, step)
        dto = UpdateDiversityFilterDTO(score_summary, [], step)
        score = self._diversity_filter.update_score(dto)
        return score, score_summary

    def _updating(self, sampled, score, inception, agent):
        agent_likelihood, prior_likelihood, augmented_likelihood = \
            self.learning_strategy.run(sampled, score, inception, agent)
        return agent_likelihood, prior_likelihood, augmented_likelihood

    def _logging(self, agent: GenerativeModelBase, start_time: float, step: int, score_summary: FinalSummary,
                  agent_likelihood: torch.tensor, prior_likelihood: torch.tensor, augmented_likelihood: torch.tensor):
        report_dto = TimestepDTO(start_time, self._parameters.max_num_iterations, step, score_summary,
                                 agent_likelihood, prior_likelihood, augmented_likelihood)
        self._logger.timestep_report(report_dto, self._diversity_filter, agent)

    def save_and_flush_memory(self, agent, memory_name: str):
        self._logger.save_merging_state(agent, self._diversity_filter, name=memory_name)
        self._diversity_filter = DiversityFilter(self._parameters.diversity_filter)
