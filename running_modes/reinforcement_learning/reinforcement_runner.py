import time

import numpy as np
import torch

from models.model import Model
from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLearningConfiguration
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.reinforcement_learning.margin_guard import MarginGuard
from running_modes.utils.general import to_tensor
from diversity_filters.base_diversity_filter import BaseDiversityFilter
from reinvent_chemistry.utils import get_indices_of_unique_smiles

from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction
from reinvent_scoring.scoring.score_summary import FinalSummary


class ReinforcementRunner:
    def __init__(self, envelope: GeneralConfigurationEnvelope, config: ReinforcementLearningConfiguration,
                 diversity_filter: BaseDiversityFilter,
                 scoring_function: BaseScoringFunction, inception: Inception):
        self._prior = Model.load_from_file(config.prior)
        self._agent = Model.load_from_file(config.agent)
        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self.config = config
        self._logger = ReinforcementLogger(envelope)
        self._inception = inception
        self._margin_guard = MarginGuard(self)
        self._optimizer = torch.optim.Adam(self._agent.network.parameters(), lr=self.config.learning_rate)

        assert self._prior.vocabulary == self._agent.vocabulary, "The agent and the prior must have the same vocabulary"

    def run(self):
        self._logger.log_message("starting an RL run")
        start_time = time.time()
        reset_countdown = 0
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

            reset_countdown = self._stats_and_chekpoint(score, start_time, step, smiles, score_summary,
                                                        agent_likelihood, prior_likelihood,
                                                        augmented_likelihood, reset_countdown)

        self._logger.save_final_state(self._agent, self._diversity_filter)
        self._logger.log_out_input_configuration()
        self._logger.log_out_inception(self._inception)

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.network.parameters():
            param.requires_grad = False

    def _stats_and_chekpoint(self, score, start_time, step, smiles, score_summary: FinalSummary,
                             agent_likelihood, prior_likelihood, augmented_likelihood, reset_countdown):
        self._margin_guard.adjust_margin(step)
        mean_score = np.mean(score)
        self._margin_guard.store_run_stats(agent_likelihood, prior_likelihood, augmented_likelihood, score)
        self._logger.timestep_report(start_time, self.config.n_steps, step, smiles,
                                     mean_score, score_summary, score,
                                     agent_likelihood, prior_likelihood, augmented_likelihood, self._diversity_filter)
        self._logger.save_checkpoint(step, self._diversity_filter, self._agent)
        return self._update_reset_countdown(reset_countdown, mean_score)

    def _sample_unique_sequences(self, agent, batch_size):
        seqs, smiles, agent_likelihood = agent.sample_sequences_and_smiles(batch_size)
        unique_idxs = get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique

    def _update_reset_countdown(self, reset_countdown, mean_score):
        """reset the weight of NN to search for diverse solutions"""
        if self.config.reset:
            if reset_countdown:
                reset_countdown += 1
            elif mean_score >= self.config.reset_score_cutoff:
                reset_countdown = 1

            if reset_countdown == self.config.reset:
                reset_countdown = self.reset()
        return reset_countdown

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

    def reset(self, reset_countdown=0):
        self._agent = Model.load_from_file(self.config.agent)
        self._optimizer = torch.optim.Adam(self._agent.network.parameters(), lr=self.config.learning_rate)
        self._logger.log_message("Resetting Agent")
        self._logger.log_message(f"Adjusting sigma to: {self.config.sigma}")
        return reset_countdown
