from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import numpy as np
import torch
from reinvent_chemistry import get_indices_of_unique_smiles
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring import ScoringFuncionParameters, ScoringFunctionFactory, FinalSummary
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction
from running_modes.enums.scoring_table_enum import ScoringTableEnum
from running_modes.enums.special_parameters_enum import SpecialParametersEnum

from running_modes.automated_curriculum_learning.logging.base_auto_cl_logger import BaseAutoCLLogger
from running_modes.automated_curriculum_learning.scoring_table import ScoringTable
from running_modes.configurations.automated_curriculum_learning.ranking_strategy_configuration import \
    RankingStrategyConfiguration
from running_modes.utils import to_tensor

from running_modes.reinforcement_learning.inception import Inception


class BaseRankingStrategy(ABC):
    def __init__(self, prior: Model, agent: Model, scoring_function_params: ScoringFuncionParameters,
                 diversity_filter: BaseDiversityFilter, inception: Inception,
                 configuration: RankingStrategyConfiguration, logger: BaseAutoCLLogger, scoring_table: ScoringTable):
        self._agent = agent
        self._prior = prior
        self._special_parameters_enum = SpecialParametersEnum()
        self._parameters = configuration.parameters
        self._scoring_function_name = scoring_function_params.name
        # List of scoring function components
        self._scoring_function_params = scoring_function_params.parameters
        self._diversity_filter = diversity_filter
        self._inception = inception
        self._scoring_table = scoring_table
        self._scoring_table_enum = ScoringTableEnum()
        self._logger = logger

    def _setup_scoring_function(self, name: str, parameter_list: List[dict]) -> BaseScoringFunction:
        scoring_function_parameters = ScoringFuncionParameters(name=name, parameters=parameter_list, parallel=False)
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        return scoring_function_instance

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

    @abstractmethod
    def train_agent(self, agent: Model, scoring_function_component: Any):
        raise NotImplementedError("train_agents not implemented.")

    def rank_agents(self):
        for component in self._scoring_function_params:
            specific_params = component.get("specific_parameters", {})
            if specific_params.get("always_present", False):
                self._scoring_table.add_constant_component(component)
                self._logger.log_message(
                    f"adding constant scoring component: {component.get('component_type')}")
            else:
                agent = self._agent
                self._logger.log_message(f"ranking of agent with the scoring component: "
                                         f"{component.get('component_type')}")
                self.train_agent(agent=agent, scoring_function_component=component)

        self._logger.log_text_to_file(self._scoring_table.rank_by_score()[[self._scoring_table_enum.SCORES,
                                                                           self._scoring_table_enum.COMPONENT_NAMES]])

    def _take_step(self, agent: Model, optimiser: Any, scoring_function_current: BaseScoringFunction, step: int) \
            -> Tuple[Model, float]:

        seqs, smiles, agent_likelihood = self._sample_unique_sequences(agent=agent,
                                                                       batch_size=self._parameters.batch_size)
        agent_likelihood = -agent_likelihood
        prior_likelihood = -self._prior.likelihood(seqs)
        score_summary: FinalSummary = scoring_function_current.get_final_score_for_step(smiles, step)
        score = self._diversity_filter.update_score(score_summary, step)
        augmented_likelihood = prior_likelihood + self._parameters.sigma * to_tensor(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        loss, agent_likelihood = self._inception_filter(agent, loss, agent_likelihood, prior_likelihood,
                                                        self._parameters.sigma, smiles, score)
        score = score.mean()
        loss = loss.mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        return agent, score