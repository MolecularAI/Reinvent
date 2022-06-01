from abc import ABC, abstractmethod
from typing import Union

import torch
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import ScoringFunctionFactory, Conversions
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.dto import CurriculumOutcomeDTO
from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.configurations import CurriculumStrategyConfiguration
from running_modes.configurations.automated_curriculum_learning.curriculum_strategy_input_configuration import \
    CurriculumStrategyInputConfiguration


class BaseCurriculumStrategy(ABC):
    def __init__(self, prior: GenerativeModelBase, agent: GenerativeModelBase,
                 configuration: Union[CurriculumStrategyConfiguration, CurriculumStrategyInputConfiguration],
                 diversity_filter: BaseDiversityFilter, inception: Inception,
                 logger: BaseLogger):

        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._conversion = Conversions()
        self._parameters = configuration
        self._prior = prior
        self._agent = agent
        self._logger = logger
        optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self._parameters.learning_rate)
        self.learning_strategy = LearningStrategy(self._prior, optimizer, configuration.learning_strategy, logger)
        self._diversity_filter = diversity_filter
        self.inception = inception

    @abstractmethod
    def run(self) -> CurriculumOutcomeDTO:
        raise NotImplementedError("run() method is not implemented ")

    @abstractmethod
    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        raise NotImplementedError("take_step() method is not implemented ")

    def save_and_flush_memory(self, agent, memory_name: str):
        self._logger.save_merging_state(agent, self._diversity_filter, name=memory_name)
        self._diversity_filter = DiversityFilter(self._parameters.diversity_filter)

    def disable_prior_gradients(self):
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def _setup_scoring_function(self, item_id: int) -> BaseScoringFunction:
        parameters = self._parameters.curriculum_objectives[item_id]
        scoring_function_instance = ScoringFunctionFactory(parameters.scoring_function)
        self._logger.log_message(f"Loading a curriculum step number {item_id}")
        return scoring_function_instance

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

    def promote_agent(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                      step_counter: int, start_time: float, merging_threshold: float) -> int:

        while not self._is_step_quota_exceeded(step_counter):
            score = self.take_step(agent=agent, scoring_function=scoring_function,
                                   step=step_counter, start_time=start_time)
            if self._is_ready_to_promote(merging_threshold, score):
                break

            step_counter += 1

        return step_counter
