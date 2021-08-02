import unittest
from unittest.mock import Mock
from typing import List

import torch
import numpy as np

from running_modes.reinforcement_learning.margin_guard import MarginGuard


class MarginGuardAdjustGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_likelihood = torch.tensor([1., 2., 3.])
        self.prior_likelihood = torch.tensor([4., 5., 6.])
        self.score = np.array([1., 2., 3])
        self.runner = Mock()
        self.mg = MarginGuard(self.runner, margin_window=1)

    def _mg_runner_config(self) -> None:
        self.mg.runner.config = Mock()
        self.mg.runner.config.sigma = 0.1
        self.mg.runner.config.margin_threshold = 0.1
        self.mg.adjust_margin(1)

    def _mg_store_run_stats(self, augmented_likelihood: List[int]):
        self.mg.store_run_stats(
            self.agent_likelihood,
            self.prior_likelihood,
            torch.tensor(augmented_likelihood),
            self.score
        )

        self._mg_runner_config()

    def testNoChange(self):
        self._mg_store_run_stats([7., 8., 9.])
        self.assertEqual(self.mg.runner.config.sigma, 0.1)

    def testChange(self):
        self._mg_store_run_stats([-7., -8., -9.])
        self.assertEqual(self.mg.runner.config.sigma, 0.2)
