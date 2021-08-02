from typing import List

import torch.utils.data as tud
from reinvent_models.lib_invent.models.dataset import DecoratorDataset

from running_modes.lib_invent.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.lib_invent.rl_actions import BaseAction


class LikelihoodEvaluation(BaseAction):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of CalculateNLLsFromModel.
        :param model: A generative model instance.
        :param batch_size: Batch size to use.
        :return:
        """
        super().__init__(logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, scaffold_decoration_list: List[SampledSequencesDTO]):
        """
        Calculates the NLL for a set of SMILES strings.
        :param scaffold_decoration_list: List with pairs of (scaffold, decoration) SMILES.
        :return: A tuple that follows the same order as the input list of SampledSequencesDTO.
        """
        scaffold_decoration_list = [[ss.scaffold, ss.decoration] for ss in scaffold_decoration_list]
        dataset = DecoratorDataset(scaffold_decoration_list, self.model.vocabulary)
        dataloader = tud.DataLoader(dataset, batch_size=len(dataset), collate_fn=DecoratorDataset.collate_fn,
                                    shuffle=False)
        for scaffold_batch, decorator_batch in dataloader:
            nll = self.model.likelihood(*scaffold_batch, *decorator_batch)
            return scaffold_batch, decorator_batch, nll
