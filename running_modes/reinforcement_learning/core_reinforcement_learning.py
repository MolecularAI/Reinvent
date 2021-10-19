import time

import numpy as np
import torch
from reinvent_chemistry.utils import get_indices_of_unique_smiles
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.configurations import ReinforcementLearningConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.reinforcement_learning.margin_guard import MarginGuard
from running_modes.utils.general import to_tensor


class CoreReinforcementRunner(BaseRunningMode):

    def __init__(self, critic: GenerativeModelBase, actor: GenerativeModelBase,
                 configuration: ReinforcementLearningConfiguration,
                 scoring_function: BaseScoringFunction, diversity_filter: BaseDiversityFilter,
                 inception: Inception, logger: BaseReinforcementLogger):
        self._prior = critic
        self._agent = actor
        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self.config = configuration
        self._logger = logger
        self._inception = inception
        self._margin_guard = MarginGuard(self)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)

    def run(self):
        self._logger.log_message("starting an RL run")
        start_time = time.time()
        self._disable_prior_gradients()

        for step in range(self.config.n_steps):
            seqs, smiles, agent_likelihood = self._sample_unique_sequences(self._agent, self.config.batch_size)
            # switch signs
            agent_likelihood = -agent_likelihood
            prior_likelihood = -self._prior.likelihood(seqs)
            score_summary: FinalSummary = self._scoring_function.get_final_score_for_step(smiles, step)
            score = self._diversity_filter.update_score(score_summary, step)

            augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(score)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            loss, agent_likelihood = self._inception_filter(self._agent, loss, agent_likelihood, prior_likelihood,
                                                            self.config.sigma, smiles, score)
            loss = loss.mean()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._stats_and_chekpoint(score, start_time, step, smiles, score_summary,
                                                        agent_likelihood, prior_likelihood,
                                                        augmented_likelihood)

        self._logger.save_final_state(self._agent, self._diversity_filter)
        self._logger.log_out_input_configuration()
        self._logger.log_out_inception(self._inception)

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def _stats_and_chekpoint(self, score, start_time, step, smiles, score_summary: FinalSummary,
                             agent_likelihood, prior_likelihood, augmented_likelihood):
        self._margin_guard.adjust_margin(step)
        mean_score = np.mean(score)
        self._margin_guard.store_run_stats(agent_likelihood, prior_likelihood, augmented_likelihood, score)
        self._logger.timestep_report(start_time, self.config.n_steps, step, smiles,
                                     mean_score, score_summary, score,
                                     agent_likelihood, prior_likelihood, augmented_likelihood, self._diversity_filter)
        self._logger.save_checkpoint(step, self._diversity_filter, self._agent)

    def _sample_unique_sequences(self, agent, batch_size):
        seqs, smiles, agent_likelihood = agent.sample(batch_size)
        unique_idxs = get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood, sigma, smiles, score):
        exp_smiles, exp_scores, exp_prior_likelihood = self._inception.sample()
        if len(exp_smiles) > 0:
            exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_scores
            exp_loss = torch.pow((to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
        self._inception.add(smiles, score, prior_likelihood)

        return loss, agent_likelihood

    def reset(self, reset_countdown=0):
        model_type_enum = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()
        actor_config = ModelConfiguration(model_type_enum.DEFAULT, model_regime.TRAINING,
                                          self.config.agent)
        self._agent = GenerativeModel(actor_config)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)
        self._logger.log_message("Resetting Agent")
        self._logger.log_message(f"Adjusting sigma to: {self.config.sigma}")
        return reset_countdown
