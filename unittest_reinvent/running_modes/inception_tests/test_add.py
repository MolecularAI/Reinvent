from unittest_reinvent.running_modes.inception_tests.test_add_base import TestInceptionAddBase
from unittest_reinvent.fixtures.test_data import ETHANE, BUTANE, PROPANE
from typing import List

import numpy as np
import torch as ts


class TestInceptionModelAdd(TestInceptionAddBase):
    def setUp(self):
        super().setUp()

        self.smiles = [PROPANE, BUTANE, ETHANE]
        self.score = [0, 0.5, 1.]
        self.prior_likelihood = [0, 10, 100]

    def _add_smiles(self, smiles: List[str], score: List[int], likelihood: List[int]):
        smiles = np.array(smiles)
        prior_likelihood = ts.tensor(likelihood)
        self.inception_model.add(smiles, score, prior_likelihood)

    def test_inception_model_1(self):
        self.assertEqual(len(self.inception_model.memory), 3)

    def test_inception_model_2(self):
        self._add_smiles(self.smiles, self.score, self.prior_likelihood)
        self.assertEqual(len(self.inception_model.memory), 4)

    def test_inception_model_3(self):
        self._add_smiles(self.smiles, self.score, self.prior_likelihood)
        self._add_smiles(self.smiles, self.score, self.prior_likelihood)
        self.assertEqual(len(self.inception_model.memory), 4)

    def test_inception_model_4(self):
        self._add_smiles(self.smiles, self.score, self.prior_likelihood)
        exp_seqs, exp_score, exp_prior_likelihood = self.inception_model.sample()
        self.assertEqual(len(exp_seqs), 4)

    def test_inception_model_5(self):
        self._add_smiles(self.smiles, self.score, self.prior_likelihood)
        exp_seqs, exp_score, exp_prior_likelihood = self.inception_model.sample()
        self.assertEqual(len(exp_seqs), self.inception_model.configuration.memory_size)
        self.assertEqual(len(exp_score), self.inception_model.configuration.sample_size)
        self.assertEqual(len(exp_prior_likelihood), len(exp_seqs))
        self.assertEqual(len(exp_prior_likelihood), len(exp_score))

    def test_inception_model_6(self):
        self._add_smiles(self.smiles, self.score, self.prior_likelihood)
        len0 = len(self.inception_model.sample()[0])
        len1 = len(self.inception_model.sample()[1])
        len2 = len(self.inception_model.sample()[2])
        self._add_smiles(smiles=[PROPANE, BUTANE], score=[0, 0.5], likelihood=[0, 10])
        self.assertEqual(len(self.inception_model.sample()[0]), len0)
        self.assertEqual(len(self.inception_model.sample()[0]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[1]), len1)
        self.assertEqual(len(self.inception_model.sample()[1]), self.inception_model.configuration.sample_size)
        self.assertEqual(len(self.inception_model.sample()[2]), len2)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.sample_size)
