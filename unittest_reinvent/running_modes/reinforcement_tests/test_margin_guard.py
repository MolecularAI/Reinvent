import unittest
from unittest.mock import Mock

import torch
import numpy as np

from running_modes.reinforcement_learning.margin_guard import MarginGuard


class MarginGuardStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.runner = Mock()
        self.mg = MarginGuard(self.runner)
        self.agent_likelihood = torch.tensor([[1., -1.], [1., -1.]])
        self.prior_likelihood = torch.tensor([[1., -1.], [1., -1.]])
        self.augmented_likelihood = torch.tensor([[1., -1.], [1., -1.]])
        self.score = np.array([1., 2., 3])

    def _store_run(self) -> None:
        self.mg.store_run_stats(
            self.agent_likelihood,
            self.prior_likelihood,
            self.augmented_likelihood,
            self.score
        )

    def test_empty(self):
        self.assertEqual(len(self.mg._run_stats), 0)

    def test_store_one(self):
        self._store_run()

        self.assertEqual(len(self.mg._run_stats), 1)

    def test_store_two(self):
        self._store_run()
        self._store_run()

        self.assertEqual(len(self.mg._run_stats), 2)

    def test_stats_have_all_fields(self):
        self._store_run()

        fields = {
            "agent_likelihood",
            "prior_likelihood",
            "augmented_likelihood",
            "score"
        }

        self.assertTrue(all(f in line for line in self.mg._run_stats for f in fields))


