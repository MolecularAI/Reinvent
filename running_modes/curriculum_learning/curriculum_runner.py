import time

import numpy as np
import torch
from reinvent_chemistry.utils import get_indices_of_unique_smiles
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters
from reinvent_scoring.scoring.score_summary import FinalSummary
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.configurations import GeneralConfigurationEnvelope, InceptionConfiguration
from running_modes.configurations.curriculum_learning.curriculum_learning_components import CurriculumLearningComponents
from running_modes.configurations.curriculum_learning.curriculum_learning_configuration import \
    CurriculumLearningConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.curriculum_learning.logging import CurriculumLogger
from running_modes.curriculum_learning.update_watcher import UpdateWatcher
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.reinforcement_learning.margin_guard import MarginGuard
from running_modes.utils import to_tensor


class CurriculumRunner(BaseRunningMode):
    def __init__(self, envelope: GeneralConfigurationEnvelope):
        self.envelope = envelope
        config_components = CurriculumLearningComponents(**self.envelope.parameters)
        self.config = CurriculumLearningConfiguration(**config_components.curriculum_learning)

        model_regime = GenerativeModelRegimeEnum()
        prior_config = ModelConfiguration(ModelTypeEnum().DEFAULT, model_regime.INFERENCE, self.config.prior)
        agent_config = ModelConfiguration(ModelTypeEnum().DEFAULT, model_regime.TRAINING, self.config.agent)
        _prior = GenerativeModel(prior_config)
        _agent = GenerativeModel(agent_config)

        self._prior = _prior
        self._agent = _agent
        self.logger = CurriculumLogger(self.envelope)
        self.scoring_function = self.setup_scoring_function(config_components.scoring_function)
        self.diversity_filter = self._setup_diversity_filter(config_components.diversity_filter)
        self.inception = self.setup_inception(config_components.inception)
        self._margin_guard = MarginGuard(self)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)

        self._update_watcher = UpdateWatcher(self)

    def run(self):
        self.logger.log_message("starting a Curriculum Learning run")
        self.logger.log_out_input_configuration(self.envelope)
        start_time = time.time()
        reset_countdown = 0
        self._disable_prior_gradients()
        step = 0

        while step < self.config.n_steps:
            seqs, smiles, agent_likelihood = self._sample_unique_sequences(self._agent, self.config.batch_size)
            # switch signs
            agent_likelihood = -agent_likelihood
            prior_likelihood = -self._prior.likelihood(seqs)
            score_summary: FinalSummary = self.scoring_function.get_final_score_for_step(smiles, step)
            score = self.diversity_filter.update_score(score_summary, step)

            distance_penalty = self._margin_guard.get_distance_to_prior(prior_likelihood, self.config.distance_threshold)
            score = score * distance_penalty

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

            self._update_watcher.check_for_scheduled_update(step)
            self._update_watcher.check_for_pause()
            self._update_watcher.check_for_update(step)
            step += 1

        self.logger.save_final_state(self._agent, self.diversity_filter)
        self.logger.log_out_inception(self.inception)

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def _stats_and_chekpoint(self, score, start_time, step, smiles, score_summary: FinalSummary,
                             agent_likelihood, prior_likelihood, augmented_likelihood, reset_countdown):
        self._margin_guard.adjust_margin(step)
        mean_score = np.mean(score)
        self._margin_guard.store_run_stats(agent_likelihood, prior_likelihood, augmented_likelihood, score)
        self.logger.timestep_report(start_time, self.config.n_steps, step, smiles, mean_score, score_summary, score,
                                    agent_likelihood, prior_likelihood, augmented_likelihood, self.diversity_filter)
        self.logger.save_checkpoint(step, self.diversity_filter, self._agent)
        return self._update_reset_countdown(reset_countdown, mean_score)

    def _sample_unique_sequences(self, agent, batch_size):
        seqs, smiles, agent_likelihood = agent.sample(batch_size)
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
        if self.inception is not None:
            exp_smiles, exp_scores, exp_prior_likelihood = self.inception.sample()
            distance_penalty = self._margin_guard.get_distance_to_prior(prior_likelihood,
                                                                        self.config.distance_threshold)
            score = score * distance_penalty

            if len(exp_smiles) > 0:
                exp_distance_penalty = self._margin_guard.get_distance_to_prior(exp_prior_likelihood,
                                                                                   self.config.distance_threshold)
                exp_scores = np.array(exp_scores) * exp_distance_penalty
                exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
                exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_scores
                exp_loss = torch.pow((to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            self.inception.add(smiles, score, prior_likelihood)
        return loss, agent_likelihood

    def reset(self, reset_countdown=0):
        model_regime = GenerativeModelRegimeEnum()
        agent_config = ModelConfiguration(self.envelope.model_type, model_regime.TRAINING, self.config.agent)

        self._agent = GenerativeModel(agent_config)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)
        self.logger.log_message("Resetting Agent")
        self.logger.log_message(f"Adjusting sigma to: {self.config.sigma}")
        return reset_countdown

    def _setup_diversity_filter(self, diversity_filter_parameters):
        diversity_filter_parameters = DiversityFilterParameters(**diversity_filter_parameters)
        diversity_filter = DiversityFilter(diversity_filter_parameters)
        return diversity_filter

    def setup_scoring_function(self, scoring_function_parameters):
        scoring_function_parameters = ScoringFunctionParameters(**scoring_function_parameters)
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        return scoring_function_instance

    def setup_inception(self, inception_parameters):
        inception_config = InceptionConfiguration(**inception_parameters)
        inception = Inception(inception_config, self.scoring_function, self._prior)
        return inception

