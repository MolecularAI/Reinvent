from abc import ABC, abstractmethod
from typing import Tuple, List, Any

import numpy as np
import torch
from reinvent_chemistry import get_indices_of_unique_smiles
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFunctionFactory, FinalSummary
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.configurations import CurriculumStrategyConfiguration
from running_modes.reinforcement_learning.inception import Inception
from running_modes.utils import to_tensor


class BaseCurriculumStrategy(ABC):
    def __init__(self, prior: Model, agent: Model, diversity_filter: BaseDiversityFilter,
                 inception: Inception, configuration: CurriculumStrategyConfiguration,
                 logger: BaseAutoCLLogger):
        self._parameters = configuration
        self._prior = prior
        self._agent = agent
        self._diversity_filter = diversity_filter
        self.inception = inception
        self._logger = logger
        self._max_num_iterations = configuration.max_num_iterations

    @abstractmethod
    def run(self) -> Tuple[Model, int]:
        raise NotImplementedError("run not implemented for the current merging strategy")

    def _log_sf_update(self, current_parameters: List[dict], step: int):
        text_to_log = f" Merging  \n Step: {step}, #SF parameters: {len(current_parameters)} " \
                      f"\n used components: {[component.get('name') for component in current_parameters]}"
        self._logger.log_message(text_to_log)

    def _sample_unique_sequences(self, agent: Model, batch_size: int) -> Tuple[Any, Any, Any]:
        seqs, smiles, agent_likelihood = agent.sample_sequences_and_smiles(batch_size)
        unique_idxs = get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique

    def disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.network.parameters():
            param.requires_grad = False

    def save_and_flush_memory(self, agent, memory_name: str):
        self._logger.save_merging_state(agent, self._diversity_filter, name=memory_name)
        self._diversity_filter = DiversityFilter(self._parameters.diversity_filter)

    def _stats_and_chekpoint(self, agent: Model, score: Any, start_time: float, step: int, smiles: List,
                             score_summary: FinalSummary, agent_likelihood: torch.tensor,
                             prior_likelihood: torch.tensor,
                             augmented_likelihood: torch.tensor):
        mean_score = np.mean(score)
        self._logger.timestep_report(start_time, self._max_num_iterations, step, smiles,
                                     mean_score, score_summary, score,
                                     agent_likelihood, prior_likelihood, augmented_likelihood, self._diversity_filter)
        self._logger.save_checkpoint(step, self._diversity_filter, agent)

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood,
                          sigma, smiles, score):
        if self.inception:
            exp_smiles, exp_scores, exp_prior_likelihood = self.inception.sample()
            if len(exp_smiles) > 0:
                exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
                exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_scores
                exp_loss = torch.pow((to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            self.inception.add(smiles, score, prior_likelihood)
        return loss, agent_likelihood

    def _is_ready_to_promote(self, promotion_threshold: float, score: float) -> bool:
        if score >= promotion_threshold:
            message = f"** Promotion condition reached: Threshold. Score: {score} >= threshold: {promotion_threshold}"
            self._logger.log_message(message)
            return True
        return False

    def _is_step_quota_exceeded(self, current_step: int) -> bool:
        if self._parameters.max_num_iterations <= current_step:
            message = f"** The delegated step quota for training is exceeded: {current_step}"
            self._logger.log_message(message)
            return True
        return False

    def _setup_scoring_function(self, item_id: int) -> BaseScoringFunction:
        parameters = self._parameters.curriculum_objectives[item_id]
        scoring_function_instance = ScoringFunctionFactory(parameters.scoring_function)
        self._logger.log_message(f"Loading a curriculum step number {item_id}")
        return scoring_function_instance

    def take_step(self, agent: Model, optimiser: Any, scoring_function_current: BaseScoringFunction, step: int,
                  start_time: float) -> float:
        seqs, smiles, agent_likelihood = self._sample_unique_sequences(agent, self._parameters.batch_size)
        agent_likelihood = -agent_likelihood
        prior_likelihood = -self._prior.likelihood(seqs)
        score_summary_current: FinalSummary = scoring_function_current.get_final_score_for_step(smiles, step)
        score = self._diversity_filter.update_score(score_summary_current, step)
        augmented_likelihood = prior_likelihood + self._parameters.sigma * to_tensor(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        loss, agent_likelihood = self._inception_filter(agent, loss, agent_likelihood, prior_likelihood,
                                                        self._parameters.sigma, smiles, score)

        loss = loss.mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        self._stats_and_chekpoint(agent=agent, score=score, start_time=start_time, step=step, smiles=smiles,
                                  score_summary=score_summary_current, agent_likelihood=agent_likelihood,
                                  prior_likelihood=prior_likelihood, augmented_likelihood=augmented_likelihood)

        score = score.mean()
        return score

    def promote_agent(self, agent: Model, optimiser: Any, scoring_function: BaseScoringFunction, step_counter: int,
                      start_time: float, merging_threshold: float) -> int:

        while not self._is_step_quota_exceeded(step_counter):
            score = self.take_step(agent=agent, optimiser=optimiser, scoring_function_current=scoring_function,
                                   step=step_counter, start_time=start_time)
            if self._is_ready_to_promote(merging_threshold, score):
                break

            step_counter += 1

        return step_counter
