from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from reinvent_chemistry import get_indices_of_unique_smiles
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFunctionFactory, FinalSummary, ScoringFunctionParameters
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.configurations.automated_curriculum_learning.production_strategy_configuration import \
    ProductionStrategyConfiguration
from running_modes.reinforcement_learning.inception import Inception
from running_modes.utils import to_tensor


class BaseProductionStrategy(ABC):
    def __init__(self, prior: Model, diversity_filter: BaseDiversityFilter, inception: Inception,
                 scoring_function: BaseScoringFunction, configuration: ProductionStrategyConfiguration,
                 logger: BaseAutoCLLogger):
        self._parameters = configuration
        self._prior = prior
        self._diversity_filter = diversity_filter
        self._inception = inception
        self._scoring_function = scoring_function
        self._logger = logger

    @abstractmethod
    def run(self, cl_agent: Model, steps_so_far: int):
        raise NotImplementedError("run not implemented.")

    def setup_scoring_function(self, name: str, parameter_list: List[Dict]) -> BaseScoringFunction:
        scoring_function_parameters = ScoringFunctionParameters(name=name, parameters=parameter_list, parallel=False)
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)

        self._log_sf_update(current_parameters=parameter_list)
        return scoring_function_instance

    def _log_sf_update(self, current_parameters: List[dict]):
        text_to_log = f"** Production setup **\n scoring_function: " \
                      f"{[component.get('name') for component in current_parameters]}"
        # log in console
        self._logger.log_message(text_to_log)

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.network.parameters():
            param.requires_grad = False

    def _sample_unique_sequences(self, agent: Model, batch_size: int) -> Tuple[Any, Any, Any]:
        seqs, smiles, agent_likelihood = agent.sample_sequences_and_smiles(batch_size)
        unique_idxs = get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood,
                          sigma, smiles, score):
        if self._inception is not None:
            exp_smiles, exp_scores, exp_prior_likelihood = self._inception.sample()
            if len(exp_smiles) > 0:
                exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
                exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_scores
                exp_loss = torch.pow((to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            self._inception.add(smiles, score, prior_likelihood)
        return loss, agent_likelihood

    def _stats_and_chekpoint(self, agent: Model, score: Any, start_time: float, step: int, smiles: List,
                             score_summary: FinalSummary, agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                             augmented_likelihood: torch.tensor):
        mean_score = np.mean(score)
        self._logger.timestep_report(start_time, self._parameters.n_steps, step, smiles,
                                     mean_score, score_summary, score,
                                     agent_likelihood, prior_likelihood, augmented_likelihood, self._diversity_filter)
        self._logger.save_checkpoint(step, self._diversity_filter, agent)

    def _take_step(self, agent: Model, optimizer: Any, scoring_function_current: BaseScoringFunction, step: int,
                   start_time: float):
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self._stats_and_chekpoint(agent=agent, score=score, start_time=start_time, step=step, smiles=smiles,
                                  score_summary=score_summary_current, agent_likelihood=agent_likelihood,
                                  prior_likelihood=prior_likelihood, augmented_likelihood=augmented_likelihood)