import unittest
from unittest.mock import Mock

import torch
import numpy as np

from running_modes.reinforcement_learning.margin_guard import MarginGuard


class MarginGuardIncreaseSigmaTest(unittest.TestCase):
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
        self.mg.runner.config = Mock()
        self.mg.runner.config.margin_threshold = 0.1

    def test_increase_sigma(self):
        self.mg.runner.config.sigma = 0.1
        sigma = self.mg._increased_sigma()
        self.assertAlmostEqual(sigma, 0.2)

    def test_increase_sigma_2(self):
        self.mg.runner.config.sigma = 0.2
        sigma = self.mg._increased_sigma()
        self.assertAlmostEqual(sigma, 0.3)

    def test_increase_sigma_3(self):
        self.mg.runner.config.sigma = 0.01
        sigma = self.mg._increased_sigma()
        self.assertAlmostEqual(sigma, 0.11)

    def test_increase_sigma_4(self):
        self.mg.runner.config.sigma = -10
        sigma = self.mg._increased_sigma()
        self.assertAlmostEqual(sigma, -1.4)
