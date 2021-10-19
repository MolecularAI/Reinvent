import torch

from running_modes.reinforcement_learning.learning_strategy import LearningStrategyEnum
from unittest_reinvent.running_modes.lib_invent_tests.learning_strategy_tests.base_learning_strategy import \
    BaseTestLearningStrategy


class TestLearningStrategySdapStrategy(BaseTestLearningStrategy):

    def setUp(self):
        self.learning_strategy = LearningStrategyEnum().SDAP
        super().setUp()

    def test_sdap_strategy(self):
        actor_nlls, critic_nlls, augmented_nlls = \
            self.runner.run(self.scaffold_batch, self.decorator_batch, self.score, self.actor_nlls)

        self.assertEqual(actor_nlls, torch.neg(self.actor_nlls))
        self.assertEqual(critic_nlls, -0.3)
        self.assertEqual(augmented_nlls, 89.7)
