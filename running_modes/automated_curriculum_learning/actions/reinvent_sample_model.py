from typing import Tuple, Any

import numpy as np
from reinvent_chemistry import Conversions
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

from running_modes.automated_curriculum_learning.actions import BaseSampleAction
from running_modes.automated_curriculum_learning.dto import SampledBatchDTO


class ReinventSampleModel(BaseSampleAction):
    def __init__(self, model: GenerativeModelBase, batch_size: int, logger=None):
        """
        Creates an instance of SampleModel.
        :params model: A model instance.
        :params batch_size: Batch size to use.
        :return:
        """
        super().__init__(logger)
        self.model = model
        self._batch_size = batch_size

        self._conversions = Conversions()

    def run(self) -> SampledBatchDTO:
        seqs, smiles, agent_likelihood = self._sample_unique_sequences(self.model, self._batch_size)
        batch = SampledBatchDTO(seqs, smiles, agent_likelihood)

        return batch

    def _sample_unique_sequences(self, agent: GenerativeModelBase, batch_size: int) -> Tuple[Any, Any, Any]:
        seqs, smiles, agent_likelihood = agent.sample(batch_size)
        unique_idxs = self._get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique
