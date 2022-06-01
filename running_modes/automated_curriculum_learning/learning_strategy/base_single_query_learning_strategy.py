from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch

from running_modes.automated_curriculum_learning.dto import SampledBatchDTO
from running_modes.automated_curriculum_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration


class BaseSingleQueryLearningStrategy(BaseLearningStrategy):
    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger):
        super().__init__(critic_model, optimizer, configuration, logger)

    # TODO: Return the loss as well.
    def run(self, sampled: SampledBatchDTO , score: torch.Tensor, inception, agent) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
            self._calculate_loss(sampled.smiles, sampled.sequences, score, sampled.likelihoods, inception, agent)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return negative_actor_nlls, negative_critic_nlls, augmented_nlls

    @abstractmethod
    def _calculate_loss(self, smiles, sampled_sequences: np.ndarray, score, actor_nlls, inception, agent):
        raise NotImplementedError("_calculate_loss method is not implemented")
