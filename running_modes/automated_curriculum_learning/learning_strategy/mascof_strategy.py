import torch
from reinvent_models.link_invent.dto import BatchLikelihoodDTO

from running_modes.automated_curriculum_learning.dto import UpdatedLikelihoodsDTO
from running_modes.automated_curriculum_learning.learning_strategy.base_double_query_learning_strategy import \
    BaseDoubleQueryLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration


class MASCOFStrategy(BaseDoubleQueryLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)

    def _calculate_loss(self, likelihood_dto: BatchLikelihoodDTO, score) -> UpdatedLikelihoodsDTO:
        batch = likelihood_dto.batch
        critic_nlls = self.critic_model.likelihood(*batch.input, *batch.output)
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -likelihood_dto.likelihood
        augmented_nlls = self._to_tensor(score)
        loss = -torch.sum(self._to_tensor(score)) * torch.sum(negative_actor_nlls)
        dto = UpdatedLikelihoodsDTO(negative_actor_nlls, negative_critic_nlls, augmented_nlls, loss)
        return dto