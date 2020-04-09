import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils as utils_general
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from running_modes.sampling.logging.base_sampling_logger import BaseSamplingLogger
from utils.logging.tensorboard import add_mols


class LocalSamplingLogger(BaseSamplingLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._summary_writer = self._instantiate_summary_writer(configuration)

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, smiles: [], likelihoods: np.array):
        self._log_timestep(smiles, likelihoods)

    def __del__(self):
        self._summary_writer.close()

    def _log_timestep(self, smiles: np.array, likelihoods: np.array):
        fraction_valid_smiles = utils_general.fraction_valid_smiles(smiles)
        fraction_unique_entries = self._get_unique_entires_fraction(likelihoods)
        self._visualize_structures(smiles)
        self._summary_writer.add_text('Data', f'Valid SMILES: {fraction_valid_smiles}% '
                                              f'Unique Mols: {fraction_unique_entries}%  ')

    def _visualize_structures(self, smiles):
        list_of_labels, list_of_mols = self._count_unique_inchi_keys(smiles)
        if len(list_of_mols) > 0:
            add_mols(self._summary_writer, "Most Frequent Molecules", list_of_mols, self._rows, list_of_labels)

    def _instantiate_summary_writer(self, configuration):
        log_config = SamplingLoggerConfiguration(**configuration.logging)
        return SummaryWriter(log_dir=log_config.logging_path)