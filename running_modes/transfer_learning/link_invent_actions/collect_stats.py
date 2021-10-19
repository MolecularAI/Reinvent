import random
from reinvent_chemistry import TransformationTokens
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry.conversions import Conversions
from typing import List, Optional

import numpy as np
import scipy.stats as sps
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.link_invent.dto import SampledSequencesDTO

from running_modes.transfer_learning.dto.collected_stats_dto import CollectedStatsDTO
from running_modes.transfer_learning.dto.sampled_stats_dto import SampledStatsDTO
from running_modes.transfer_learning.link_invent_actions.base_action import BaseAction
from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger


class CollectStats(BaseAction):
    def __init__(self, model: GenerativeModelBase, training_data: List[List[str]],
                 validation_data: Optional[List[List[str]]], logger: BaseTransferLearningLogger, sample_size,
                 initialize_data_loader_func):

        BaseAction.__init__(self, logger=logger)

        self._model = model
        self._training_data = training_data
        self._validation_data = validation_data
        self._sample_size = sample_size
        self._get_data_loader = initialize_data_loader_func

        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._conversions = Conversions()
        self._tokens = TransformationTokens()

    def run(self) -> CollectedStatsDTO:

        self._logger.log_message("Collecting stats")

        # collect training stats
        training_data_loader = self._get_data_loader(self._get_subset(self._training_data), batch_size=128,
                                                     shuffle=False)
        training_nll_list, training_sampled_stats = self._calc_stats(training_data_loader)

        if self._validation_data is not None:
            validation_data_loader = self._get_data_loader(self._get_subset(self._validation_data), batch_size=128,
                                                           shuffle=False)
            validation_nll_list, validation_sampled_stats = self._calc_stats(validation_data_loader)
            dist = [training_sampled_stats.nll_input_sampled_target, validation_sampled_stats.nll_input_sampled_target,
                    training_nll_list, validation_nll_list]
        else:
            validation_nll_list = None
            validation_sampled_stats = None
            dist = [training_sampled_stats.nll_input_sampled_target, training_nll_list]

        stats = CollectedStatsDTO(jsd_binned=self._jsd(dist, binned=True), jsd_un_binned=self._jsd(dist, binned=False),
                                  nll=training_nll_list, training_stats=training_sampled_stats,
                                  validation_nll=validation_nll_list, validation_stats=validation_sampled_stats)
        return stats

    def _get_subset(self, data: List):
        subset = list(random.sample(data, self._sample_size))
        return subset

    def _calc_stats(self, data_loader):
        sampled_sequence_list = []
        nll_list = []
        for warhead_batch, linker_batch in data_loader:
            sampled_sequence_list += self._model.sample(*warhead_batch)
            nll_list += list(self._model.likelihood(*warhead_batch, *linker_batch).data.cpu().numpy())
        sample_stats = self._get_sampled_stats(sampled_sequence_list)
        return nll_list, sample_stats

    def _jsd(self, dists, binned=False):
        min_size = min(len(dist) for dist in dists)
        dists = [dist[:min_size] for dist in dists]
        if binned:
            dists = [self._bin_dist(dist) for dist in dists]
        num_dists = len(dists)
        avg_dist = np.sum(dists, axis=0) / num_dists
        return sum((sps.entropy(dist, avg_dist) for dist in dists)) / num_dists

    @staticmethod
    def _bin_dist(dist, bins=1000, dist_range=(0, 100)):
        bins = np.histogram(dist, bins=bins, range=dist_range, density=False)[0]
        bins[bins == 0] = 1
        return bins / bins.sum()

    def _get_sampled_stats(self, sampled_sequence_list: List[SampledSequencesDTO]) -> SampledStatsDTO:
        nll_list = []
        molecule_smiles_list = []
        molecule_parts_smiles_list = []
        for sample in sampled_sequence_list:

            nll_list.append(sample.nll)

            labeled_linker = self._attachment_points.add_attachment_point_numbers(sample.output, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations(labeled_linker, sample.input)
            molecule_smiles = self._conversions.mol_to_smiles(molecule) if molecule else None
            molecule_is_valid = True if molecule_smiles else False
            molecule_parts_smiles = sample.input + self._tokens.ATTACHMENT_SEPARATOR_TOKEN + sample.output

            if molecule_is_valid:
                molecule_smiles_list.append(molecule_smiles)
                molecule_parts_smiles_list.append(molecule_parts_smiles)

        sample_stats = SampledStatsDTO(nll_input_sampled_target=nll_list,
                                       molecule_smiles=molecule_smiles_list,
                                       molecule_parts_smiles=molecule_parts_smiles_list,
                                       valid_fraction=len(molecule_smiles_list) / len(nll_list) * 100)
        return sample_stats

