import time
from abc import abstractmethod
from typing import List, Dict, Tuple

import numpy as np
import torch
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import ScoringFunctionFactory, FinalSummary, ScoringFunctionParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.actions.reinvent_sample_model import ReinventSampleModel
from running_modes.automated_curriculum_learning.dto import SampledBatchDTO, TimestepDTO
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.automated_curriculum_learning.production_strategy.base_production_strategy import \
    BaseProductionStrategy


class ReinventProductionStrategy(BaseProductionStrategy):

    def run(self, cl_agent: GenerativeModelBase, steps_so_far: int):
        start_time = time.time()
        self._disable_prior_gradients()
        optimizer = torch.optim.Adam(cl_agent.get_network_parameters(), lr=self._parameters.learning_rate)
        learning_strategy = LearningStrategy(self._prior, optimizer, self._parameters.learning_strategy, self._logger)

        step_limit = steps_so_far + self._parameters.number_of_steps

        for step in range(steps_so_far, step_limit):
            self.take_step(agent=cl_agent, learning_strategy=learning_strategy, scoring_function=self._scoring_function,
                           step=step, start_time=start_time)

        self._logger.log_message(f"Production finished at step {step_limit}")
        self._logger.save_final_state(cl_agent, self._diversity_filter)

    def setup_scoring_function(self, name: str, parameter_list: List[Dict]) -> BaseScoringFunction:
        scoring_function_parameters = ScoringFunctionParameters(name=name, parameters=parameter_list, parallel=False)
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        self._log_sf_update(current_parameters=parameter_list)
        return scoring_function_instance

    def _disable_prior_gradients(self):
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def take_step(self, agent: GenerativeModelBase, learning_strategy, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        # 1. Sampling
        sampled = self._sampling(agent)
        # 2. Scoring
        score, score_summary = self._scoring(scoring_function, sampled.smiles, step)
        # 3. Updating
        agent_likelihood, prior_likelihood, augmented_likelihood = self._updating(sampled, score, self.inception, agent, learning_strategy)
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

    def _updating(self, sampled, score, inception, agent, learning_strategy):
        agent_likelihood, prior_likelihood, augmented_likelihood = learning_strategy.run(sampled, score, inception, agent)
        return agent_likelihood, prior_likelihood, augmented_likelihood

    def _logging(self, agent: GenerativeModelBase, start_time: float, step: int, score_summary: FinalSummary,
                  agent_likelihood: torch.tensor, prior_likelihood: torch.tensor, augmented_likelihood: torch.tensor):
        report_dto = TimestepDTO(start_time, self._parameters.number_of_steps, step, score_summary,
                                 agent_likelihood, prior_likelihood, augmented_likelihood)
        self._logger.timestep_report(report_dto, self._diversity_filter, agent)