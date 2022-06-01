from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import LearningStrategyConfiguration


class BaseLearningStrategy(ABC):
    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger):
        self.critic_model = critic_model
        self.optimizer = optimizer
        self._configuration = configuration
        self._running_mode_enum = GenerativeModelRegimeEnum()
        self._logger = logger
        self._disable_prior_gradients()

    def log_message(self, message: str):
        self._logger.log_message(message)

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

    def _to_tensor(self, array, use_cuda=True):
        if torch.cuda.is_available() and use_cuda:
            return torch.tensor(array, device=torch.device("cuda"))
        return torch.tensor(array, device=torch.device("cpu"))

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        self.critic_model.set_mode(self._running_mode_enum.INFERENCE)
        for param in self.critic_model.network.parameters():
            param.requires_grad = False
