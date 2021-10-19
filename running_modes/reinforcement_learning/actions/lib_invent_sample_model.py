from typing import List

import numpy as np
import reinvent_models.lib_invent.models.dataset as md
import torch.utils.data as tud
from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry.utils import get_indices_of_unique_smiles
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

from running_modes.reinforcement_learning.actions import BaseAction
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO


class LibInventSampleModel(BaseAction):

    def __init__(self, model: GenerativeModelBase, batch_size: int, logger=None, randomize=False, sample_uniquely=True):
        """
        Creates an instance of SampleModel.
        :params model: A model instance (better in scaffold_decorating mode).
        :params batch_size: Batch size to use.
        :return:
        """
        super().__init__(logger)
        self.model = model
        self._batch_size = batch_size
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._randomize = randomize
        self._conversions = Conversions()
        self._sample_uniquely = sample_uniquely

    def run(self, scaffold_list: List[str]) -> List[SampledSequencesDTO]:
        """
        Samples the model for the given number of SMILES.
        :params scaffold_list: A list of scaffold SMILES.
        :return: A list of SampledSequencesDTO.
        """
        scaffold_list = self._randomize_scaffolds(scaffold_list) if self._randomize else scaffold_list
        clean_scaffolds = [self._attachment_points.remove_attachment_point_numbers(scaffold) for scaffold in scaffold_list]
        dataset = md.Dataset(clean_scaffolds, self.model.get_vocabulary().scaffold_vocabulary,
                             self.model.get_vocabulary().scaffold_tokenizer)
        dataloader = tud.DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=md.Dataset.collate_fn)

        for batch in dataloader:
            sampled_sequences = []

            for _ in range(self._batch_size):
                scaffold_seqs, scaffold_seq_lengths = batch
                packed = self.model.sample(scaffold_seqs, scaffold_seq_lengths)
                for scaffold, decoration, nll in packed:
                    sampled_sequences.append(SampledSequencesDTO(scaffold, decoration, nll))

            if self._sample_uniquely:
                sampled_sequences = self._sample_unique_sequences(sampled_sequences)

            return sampled_sequences

    def _sample_unique_sequences(self, sampled_sequences: List[SampledSequencesDTO]) -> List[SampledSequencesDTO]:
        strings = ["".join([ss.input, ss.output]) for index, ss in enumerate(sampled_sequences)]
        unique_idxs = get_indices_of_unique_smiles(strings)
        sampled_sequences_np = np.array(sampled_sequences)
        unique_sampled_sequences = sampled_sequences_np[unique_idxs]
        return unique_sampled_sequences.tolist()

    def _randomize_scaffolds(self, scaffolds: List[str]):
        scaffold_mols = [self._conversions.smile_to_mol(scaffold) for scaffold in scaffolds]
        randomized = [self._bond_maker.randomize_scaffold(mol) for mol in scaffold_mols]
        return randomized
