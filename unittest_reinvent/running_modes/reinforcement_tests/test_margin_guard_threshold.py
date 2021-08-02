import unittest
from unittest.mock import Mock
from typing import List

import torch
import numpy as np

from running_modes.reinforcement_learning.margin_guard import MarginGuard


class MarginGuardThresholdTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_likelihood = torch.tensor([1., 2., 3.])
        self.prior_likelihood = torch.tensor([4., 5., 6.])
        self.score = np.array([1., 2., 3])
        self.runner = Mock()
        self.runner.config = Mock()
        self.runner.config.margin_threshold = 50
        self.mg = MarginGuard(self.runner)

    def _store_run_stats(self, augmented_likelihood: List[int]) -> None:
        self.mg.store_run_stats(
            self.agent_likelihood,
            self.prior_likelihood,
            torch.tensor(augmented_likelihood),
            self.score
        )

    def testFalse(self):
        self._store_run_stats([57., 58., 59.])
        self.assertFalse(self.mg._is_margin_below_threshold())

    def testTrue(self):
        self._store_run_stats([-7., -8., -9.])
        self.assertTrue(self.mg._is_margin_below_threshold())
