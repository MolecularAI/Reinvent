from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch

from running_modes.automated_curriculum_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration


class BaseLinkerStrategy(BaseLearningStrategy):
    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger):
        super().__init__(critic_model, optimizer, configuration, logger)

    # TODO: Return the loss as well.
    def run(self, scaffold_batch: np.ndarray, decorator_batch: np.ndarray,
            score: torch.Tensor, actor_nlls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
            self._calculate_loss(scaffold_batch, decorator_batch, score, actor_nlls)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return negative_actor_nlls, negative_critic_nlls, augmented_nlls

    @abstractmethod
    def _calculate_loss(self, scaffold_batch, decorator_batch, score, actor_nlls):
        raise NotImplementedError("_calculate_loss method is not implemented")
