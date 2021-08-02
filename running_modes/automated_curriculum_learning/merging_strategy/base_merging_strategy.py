import sys
from abc import ABC, abstractmethod
from typing import Tuple, Any

import numpy as np
import torch
from reinvent_chemistry import get_indices_of_unique_smiles
from reinvent_scoring import ScoringFuncionParameters, ScoringFunctionFactory, List, FinalSummary
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations.automated_curriculum_learning.merging_strategy_configuration import \
    MergingStrategyConfiguration
from running_modes.enums.scoring_table_enum import ScoringTableEnum
from running_modes.enums.special_parameters_enum import SpecialParametersEnum
from running_modes.utils import to_tensor

from running_modes.configurations import InceptionConfiguration
from running_modes.reinforcement_learning.inception import Inception


class BaseMergingStrategy(ABC):
    def __init__(self, prior: Model, scoring_function_name: str, diversity_filter: BaseDiversityFilter,
                 inception: Inception, configuration: MergingStrategyConfiguration,
                 logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        self._parameters = configuration.parameters
        self._prior = prior
        self._scoring_function_name = scoring_function_name
        self._diversity_filter = diversity_filter
        self._inception = inception
        self._logger = logger
        self._scoring_table_enum = ScoringTableEnum()
        self._special_parameters_enum = SpecialParametersEnum()
        self._scoring_table = scoring_table
        self._max_num_iterations = configuration.max_num_iterations

    @abstractmethod
    def run(self) -> Tuple[Model, int]:
        raise NotImplementedError("run not implemented for the current merging strategy")

    def setup_scoring_function(self, name: str, parameter_list: List[dict], step: int) -> BaseScoringFunction:
        parameter_list += [component for component in self._scoring_table.constant_component_table[
            self._scoring_table_enum.SCORING_FUNCTIONS]]
        scoring_function_parameters = ScoringFuncionParameters(name=name, parameters=parameter_list, parallel=False)
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)

        self._log_sf_update(current_parameters=parameter_list, step=step)
        return scoring_function_instance

    def _log_sf_update(self, current_parameters: List[dict], step: int):
        text_to_log = f" Merging  \n Step: {step}, #SF parameters: {len(current_parameters)} " \
                      f"\n used components: {[component.get('name') for component in current_parameters]}"
        self._logger.log_text_to_file(text_to_log)
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
        # self._diversity_filter.flush_memory()

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

    def _check_merging_conditions(self, merging_threshold: float, score: float, steps_of_component: int) -> bool:
        #TODO: define them in the base class intitializer
        condition_1 = score < merging_threshold
        condition_2 = steps_of_component < self._max_num_iterations
        condition = condition_1 and condition_2
        if not condition_1:
            text_to_log = f"** Merging condition reached: Threshold. Score: {score} >= threshold: {merging_threshold}"
            self._logger.log_text_to_file(text_to_log)
        if not condition_2:
            text_to_log = f"** Merging condition reached: Steps. Steps: {steps_of_component} >= threshold: {self._max_num_iterations}"
            self._logger.log_text_to_file(text_to_log)
        return condition

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

    def train_agent(self, agent: Model, optimiser: Any, scoring_function: BaseScoringFunction, step_counter: int,
                    start_time: float, merging_threshold: float, final_component: bool) -> Tuple[Model, int]:

        merging_condition = True
        steps_of_component = step_counter
        score = 0

        while merging_condition:
            score = self.take_step(agent=agent, optimiser=optimiser, scoring_function_current=scoring_function,
                                   step=step_counter, start_time=start_time)
            steps_of_component += 1
            step_counter += 1

            merging_condition = self._check_merging_conditions(merging_threshold, score, steps_of_component)

        # if the final helper component did not reach the target threshold after the maximum allowed epochs, stop the run
        if final_component and score < merging_threshold:
            sys.exit(f"The specified number of epochs: {self._max_num_iterations} failed to optimize the final helper component."
                     f"Stopping Automated Curriculum learning Run.")

        return agent, step_counter
