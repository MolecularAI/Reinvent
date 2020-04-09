import os
import shutil
import unittest
import torch
import numpy
import numpy.testing as nt
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, RANDOM_PRIOR_PATH
from models.model import Model
import utils as utils_general


class Test_model_functions(unittest.TestCase):

    def setUp(self):
        utils_general.set_default_device_cuda()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "new_model.ckpt")
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.model = Model.load_from_file(RANDOM_PRIOR_PATH)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_likelihoods_from_model_1(self):
        likelihoods = self.model.likelihood_smiles(["CCC", "c1ccccc1"])
        self.assertAlmostEqual(likelihoods[0].item(), 20.9116, 3)
        self.assertAlmostEqual(likelihoods[1].item(), 17.9506, 3)
        self.assertEqual(len(likelihoods), 2)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_sample_from_model_1(self):
        sample, nll = self.model.sample_smiles(num=20, batch_size=20)
        self.assertEqual(len(sample), 20)
        self.assertEqual(type(sample), list)
        self.assertEqual(len(nll), 20)
        self.assertEqual(type(nll), numpy.ndarray)

    def test_sample_from_model_2(self):
        seq, sample, nll = self.model.sample_sequences_and_smiles(batch_size=20)
        self.assertEqual(seq.shape[0], 20)
        self.assertEqual(type(seq), torch.Tensor)
        self.assertEqual(len(sample), 20)
        self.assertEqual(type(sample), list)
        self.assertEqual(len(nll), 20)
        self.assertEqual(type(nll), torch.Tensor)

    def test_model_tokens(self):
        tokens = self.model.vocabulary.tokens()
        self.assertIn('C', tokens)
        self.assertIn('O', tokens)
        self.assertIn('Cl', tokens)
        self.assertEqual(len(tokens), 34)
        self.assertEqual(type(tokens), list)

    def test_save_model(self):
        self.model.save(self.output_file)
        self.assertEqual(os.path.isfile(self.output_file), True)

    def test_likelihood_function_differences(self):
        seq, sample, nll = self.model.sample_sequences_and_smiles(batch_size=128)
        nll2 = self.model.likelihood(seq)
        nll3 = self.model.likelihood_smiles(sample)
        nt.assert_array_almost_equal(nll.detach().cpu().numpy(), nll2.detach().cpu().numpy(), 3)
        nt.assert_array_almost_equal(nll.detach().cpu().numpy(), nll3.detach().cpu().numpy(), 3)