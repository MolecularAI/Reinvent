import numpy as np
import numpy.testing as nt
import torch as ts

from unittest_reinvent.running_modes.inception_tests.test_empty_base import TestInceptionEmptyBase
from unittest_reinvent.fixtures.test_data import PROPANE, BUTANE, PENTANE, COCAINE


class TestInceptionEmptyAddSmiles(TestInceptionEmptyBase):

    def setUp(self):
        super().setUp()

        smiles = np.array([PROPANE, BUTANE, PENTANE, COCAINE])
        score = [0, 0.5, 0.5, 1]
        prior_likelihood = ts.tensor([0, 10, 10, 100])
        self.inception_model.add(smiles, score, prior_likelihood)

    def test_empty_eval_add_smiles(self):
        self.assertEqual(len(self.inception_model.sample()[0]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[1]), self.inception_model.configuration.sample_size)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.memory_size)
        self.assertEqual(len(self.inception_model.sample()[2]), self.inception_model.configuration.sample_size)
        nt.assert_almost_equal(np.array([1, 0.5, 0.5, 0]), np.array(self.inception_model.memory['score'].values))
        nt.assert_almost_equal(np.array([100, 10, 10, 0]), np.array(self.inception_model.memory['likelihood'].values))
