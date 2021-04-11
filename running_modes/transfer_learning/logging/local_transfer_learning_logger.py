import json
import os
from typing import List

from torch.utils.tensorboard import SummaryWriter

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger
from reinvent_chemistry.logging import fraction_valid_smiles, add_mols


class LocalTransferLearningLogger(BaseTransferLearningLogger):
    """Collects stats for an existing RNN model."""

    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._summary_writer = SummaryWriter(log_dir=self._log_config.logging_path)

    def __del__(self):
        self._summary_writer.close()

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def log_timestep(self, lr, epoch, sampled_smiles, sampled_nlls,
                     validation_nlls, training_nlls, jsd_data, jsd_joined_data, model, model_path):
        self.log_message(f"Collecting data for epoch {epoch}")

        if self._with_weights:
            self._weight_stats(model, epoch)
        if validation_nlls is not None:
            self._nll_stats_with_validation(sampled_nlls, validation_nlls, training_nlls, epoch, jsd_data,
                                            jsd_joined_data)
        elif validation_nlls is None:
            self._nll_stats(sampled_nlls, training_nlls, epoch, jsd_data, jsd_joined_data)
        self._valid_stats(sampled_smiles, epoch)
        self._visualize_structures(sampled_smiles, epoch)
        self._summary_writer.add_scalar("lr", lr, epoch)

    def _valid_stats(self, smiles, epoch):
        self._summary_writer.add_scalar("valid", fraction_valid_smiles(smiles), epoch)

    def _weight_stats(self, model, epoch):
        for name, weights in model.network.named_parameters():
            self._summary_writer.add_histogram(f"weights/{name}", weights.clone().cpu().data.numpy(), epoch)

    def _nll_stats_with_validation(self, sampled_nlls, validation_nlls, training_nlls, epoch, jsd_data,
                                   jsd_joined_data):
        self._summary_writer.add_histogram("nll_plot/sampled", sampled_nlls, epoch)
        self._summary_writer.add_histogram("nll_plot/validation", validation_nlls, epoch)
        self._summary_writer.add_histogram("nll_plot/training", training_nlls, epoch)

        self._summary_writer.add_scalars("nll/avg", {
            "sampled": sampled_nlls.mean(),
            "validation": validation_nlls.mean(),
            "training": training_nlls.mean()
        }, epoch)

        self._summary_writer.add_scalars("nll/var", {
            "sampled": sampled_nlls.var(),
            "validation": validation_nlls.var(),
            "training": training_nlls.var()
        }, epoch)

        self._summary_writer.add_scalars("nll_plot/jsd", jsd_data, epoch)
        self._summary_writer.add_scalar("nll_plot/jsd_joined", jsd_joined_data, epoch)

    def _nll_stats(self, sampled_nlls, training_nlls, epoch, jsd_data, jsd_joined_data):
        self._summary_writer.add_histogram("nll_plot/sampled", sampled_nlls, epoch)
        self._summary_writer.add_histogram("nll_plot/training", training_nlls, epoch)

        self._summary_writer.add_scalars("nll/avg", {
            "sampled": sampled_nlls.mean(),
            "training": training_nlls.mean()
        }, epoch)

        self._summary_writer.add_scalars("nll/var", {
            "sampled": sampled_nlls.var(),
            "training": training_nlls.var()
        }, epoch)

        self._summary_writer.add_scalars("nll_plot/jsd", jsd_data, epoch)
        self._summary_writer.add_scalar("nll_plot/jsd_joined", jsd_joined_data, epoch)

    def _visualize_structures(self, smiles: List[str], epoch: int):
        list_of_labels, list_of_mols = self._count_compound_frequency(smiles)
        if len(list_of_mols) > 0:
            add_mols(self._summary_writer, "Most Frequent Molecules", list_of_mols, self._rows, list_of_labels,
                     global_step=epoch)
