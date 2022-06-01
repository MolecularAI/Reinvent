import numpy as np
import torch

from running_modes.automated_curriculum_learning.learning_strategy.base_single_query_learning_strategy import \
    BaseSingleQueryLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration


class DAPSingleQueryStrategy(BaseSingleQueryLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)

        self._sigma = self._configuration.parameters.get("sigma", 120)

    def _calculate_loss(self, smiles, sampled_sequences: np.ndarray, score, actor_nlls, inception, agent):
        critic_nlls = self.critic_model.likelihood(sampled_sequences)
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -actor_nlls
        augmented_nlls = negative_critic_nlls + self._sigma * self._to_tensor(score)
        loss = torch.pow((augmented_nlls - negative_actor_nlls), 2)
        loss, agent_likelihood = self._inception_filter(agent, loss, negative_actor_nlls, negative_critic_nlls,
                                                        self._sigma, smiles, score, inception)
        loss = loss.mean()
        return loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood, sigma, smiles, score, inception):
        if inception:
            exp_smiles, exp_scores, exp_prior_likelihood = inception.sample()
            if len(exp_smiles) > 0:
                exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
                exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_scores
                exp_loss = torch.pow((self._to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            inception.add(smiles, score, prior_likelihood)
        return loss, agent_likelihood
