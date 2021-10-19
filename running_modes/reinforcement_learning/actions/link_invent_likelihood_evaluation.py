from typing import List

import torch.utils.data as tud
from reinvent_models.lib_invent.models.dataset import DecoratorDataset
from reinvent_models.link_invent.dataset.paired_dataset import PairedDataset
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.reinforcement_learning.actions import BaseAction


class LinkInventLikelihoodEvaluation(BaseAction):

    def __init__(self, model: GenerativeModelBase, logger=None):
        """
        Creates an instance of CalculateNLLsFromModel.
        :param model: A generative model instance.
        """
        super().__init__(logger)
        self.model = model

    def run(self, sampled_sequence_list: List[SampledSequencesDTO]):
        """
        Calculates the NLL for a set of SMILES strings.
        :param sampled_sequence_list: List wof sampled sequences (dto).
        :return: A tuple that follows the same order as the input list of SampledSequencesDTO.
        """
        input_output_list = [[ss.input, ss.output] for ss in sampled_sequence_list]
        dataset = PairedDataset(input_output_list, self.model.get_vocabulary())
        dataloader = tud.DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn,
                                    shuffle=False)
        for input_batch, output_batch in dataloader:
            nll = self.model.likelihood(*input_batch, *output_batch)
            return input_batch, output_batch, nll
