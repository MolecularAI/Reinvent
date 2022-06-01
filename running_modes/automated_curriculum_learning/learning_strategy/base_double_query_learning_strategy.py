from abc import abstractmethod

from reinvent_models.link_invent.dto.batch_likelihood_dto import BatchLikelihoodDTO

from running_modes.automated_curriculum_learning.dto import UpdatedLikelihoodsDTO
from running_modes.automated_curriculum_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration


class BaseDoubleQueryLearningStrategy(BaseLearningStrategy):
    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger):
        super().__init__(critic_model, optimizer, configuration, logger)

    def run(self, likelihood_dto: BatchLikelihoodDTO, score)  -> UpdatedLikelihoodsDTO:
        dto = self._calculate_loss(likelihood_dto, score)
        self.optimizer.zero_grad()
        dto.loss.backward()

        self.optimizer.step()
        return dto

    @abstractmethod
    def _calculate_loss(self, likelihood_dto: BatchLikelihoodDTO, score) -> UpdatedLikelihoodsDTO:
        raise NotImplementedError("_calculate_loss method is not implemented")
