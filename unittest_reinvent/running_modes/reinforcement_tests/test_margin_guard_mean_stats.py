import unittest
from unittest.mock import Mock

import torch
import numpy as np

from running_modes.reinforcement_learning.margin_guard import MarginGuard


class MarginGuardMeanStatsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = Mock()
        self.mg = MarginGuard(self.runner)
        self.agent_likelihood = torch.tensor([1., 2., 3.])
        self.prior_likelihood = torch.tensor([4., 5., 6.])
        self.augmented_likelihood = torch.tensor([7., 8., 9.])
        self.score = np.array([1., 2., 3])
        self.mg.store_run_stats(
            self.agent_likelihood,
            self.prior_likelihood,
            self.augmented_likelihood,
            self.score
        )

    def test_expected_mean(self):
        mean_aug_lh = self.mg._get_mean_stats_field("augmented_likelihood")
        expected_mean = self.augmented_likelihood.mean().item()
        self.assertEqual(mean_aug_lh, expected_mean)
