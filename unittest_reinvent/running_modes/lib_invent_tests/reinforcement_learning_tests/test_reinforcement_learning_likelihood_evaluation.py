import unittest

import reinvent_models.lib_invent.models.dataset as md
import torch.utils.data as tud
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.lib_invent.models.model import DecoratorModel

from unittest_reinvent.fixtures.paths import LIBINVENT_PRIOR_PATH  # TODO: Add this
from unittest_reinvent.fixtures.test_data import SCAFFOLD_SUZUKI


class TestReinforcementLearningLikelihoodEvaluation(unittest.TestCase):

    def setUp(self):

        input_scaffold = SCAFFOLD_SUZUKI

        scaffold_list_1 = [input_scaffold]
        scaffold_list_2 = [input_scaffold, input_scaffold]
        scaffold_list_3 = [input_scaffold, input_scaffold, input_scaffold]
        self._model_regime = GenerativeModelRegimeEnum()
        self.decorator_model = DecoratorModel.load_from_file(LIBINVENT_PRIOR_PATH, mode=self._model_regime.INFERENCE)

        dataset_1 = md.Dataset(scaffold_list_1 * 2, self.decorator_model.vocabulary.scaffold_vocabulary,
                               self.decorator_model.vocabulary.scaffold_tokenizer)
        self.dataloader_1 = tud.DataLoader(dataset_1, batch_size=32, shuffle=False, collate_fn=md.Dataset.collate_fn)

        dataset_2 = md.Dataset(scaffold_list_2, self.decorator_model.vocabulary.scaffold_vocabulary,
                               self.decorator_model.vocabulary.scaffold_tokenizer)
        self.dataloader_2 = tud.DataLoader(dataset_2, batch_size=32, shuffle=False, collate_fn=md.Dataset.collate_fn)

        dataset_3 = md.Dataset(scaffold_list_3, self.decorator_model.vocabulary.scaffold_vocabulary,
                               self.decorator_model.vocabulary.scaffold_tokenizer)
        self.dataloader_3 = tud.DataLoader(dataset_3, batch_size=32, shuffle=False, collate_fn=md.Dataset.collate_fn)

    def _test_scaffold_input(self, scaffold_input, expected_value):
        results = []
        for batch in scaffold_input:
            for scaff, decorations, nll in self.decorator_model.sample_decorations(*batch):
                results.append(decorations)
        self.assertEqual(expected_value, len(results))

    def test_single_scaffold_input(self):
        self._test_scaffold_input(self.dataloader_1, 2)

    def test_double_scaffold_input(self):
        self._test_scaffold_input(self.dataloader_2, 2)

    def test_triple_scaffold_input(self):
        self._test_scaffold_input(self.dataloader_3, 3)

