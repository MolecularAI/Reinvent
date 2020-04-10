import numpy as np
import scipy.stats as sps
import torch

from ...models import dataset as md
from ...models import model as mm
from ...utils import smiles as chem_smiles
from ..configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ..configurations.transfer_learning.adaptive_learning_rate_configuration import \
    AdaptiveLearningRateConfiguration
from .logging.transfer_learning_logger import TransferLearningLogger
from ...utils.enums.adaptive_learning_rate_enum import AdaptiveLearningRateEnum


class AdaptiveLearningRate:
    def __init__(self, model: mm.Model, main_config: GeneralConfigurationEnvelope,
                 configuration: AdaptiveLearningRateConfiguration):
        self._adaptive_learning_rate_enum = AdaptiveLearningRateEnum()
        self._config = configuration
        self._optimizer = torch.optim.Adam(model.network.parameters(), lr=self._config.start)
        self._learning_rate_restarted_times = 0
        self._logger = TransferLearningLogger(main_config)
        self._lr_scheduler = self._initialize_lr_scheduler()
        self._lr_adaptative_metric = []
        self._data = {}

    def _initialize_lr_scheduler(self):
        if self._config.mode == self._adaptive_learning_rate_enum.EXPONENTIAL:
            self._logger.log_message(f"Using exponential learning rate decay (gamma={self._config.gamma}, "
                                     f"step={self._config.step})")
            return torch.optim.lr_scheduler.StepLR(
                self._optimizer, step_size=self._config.step, gamma=self._config.gamma)

        elif self._config.mode == self._adaptive_learning_rate_enum.ADAPTIVE:
            self._logger.log_message(f"Using adaptative learning rate decay (gamma={self._config.gamma}, "
                                     f"threshold={self._config.threshold}, avg={self._config.average_steps})")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer, mode="min", factor=self._config.gamma, patience=self._config.patience,
                threshold=self._config.threshold)
        else:
            return None

    def update_lr_scheduler(self, epoch):
        if self._config.mode == self._adaptive_learning_rate_enum.EXPONENTIAL:
            self._lr_scheduler.step(epoch=epoch)
        if self._config.mode == self._adaptive_learning_rate_enum.ADAPTIVE:
            metric = np.mean(self._lr_adaptative_metric[-self._config.average_steps:])
            self._lr_scheduler.step(metric, epoch=epoch)

        if self.get_lr() <= self._config.restart_value and self._config.restart_times > self._learning_rate_restarted_times:
            self._logger.log_message(f"Learning rate restarted ({self._config.restart_times}): {self.get_lr()} "
                                     f"-> {self._config.start}")
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self._config.start
            self._learning_rate_restarted_times += 1

    def get_lr(self):
        return self._optimizer.param_groups[0]["lr"]

    def get_jsd_joined_data(self):
        return self._data["jsd_joined"]

    def get_jsd_data(self):
        return self._data["jsd"]

    def learning_rate_is_valid(self):
        return self.get_lr() >= self._config.min

    def clear_gradient(self):
        self._optimizer.zero_grad()

    def optimizer_step(self):
        self._optimizer.step()

    def collect_stats(self, epoch, model_path, training_set_path, validation_set_path=None):
        model = mm.Model.load_from_file(model_path, sampling_mode=True)
        training_nlls = self._calc_nlls(model, training_set_path, self._config.sample_size)
        training_nlls = self._amplify_dataset(training_nlls, self._config.sample_size)
        sampled_smiles, sampled_nlls = self._sample_smiles_and_calculate_loss(model, self._config.sample_size)

        if validation_set_path:
            validation_nlls = self._calc_nlls(model, validation_set_path, self._config.sample_size)
            validation_nlls = self._amplify_dataset(validation_nlls, self._config.sample_size)
            self._update_nll_with_validation(sampled_nlls, validation_nlls, training_nlls)
        else:
            validation_nlls = None
            self._update_nll(sampled_nlls=sampled_nlls, training_nlls=training_nlls)
        self._logger.log_timestep(lr=self.get_lr(), epoch=epoch,
                                  sampled_smiles=sampled_smiles,
                                  sampled_nlls=sampled_nlls, validation_nlls=validation_nlls,
                                  training_nlls=training_nlls,
                                  jsd_data=self.get_jsd_data(),
                                  jsd_joined_data=self.get_jsd_joined_data(), model=model)
        self._lr_adaptative_metric.append(self.get_jsd_joined_data())

    def _smiles_to_mols(self, smiles):
        smiles_and_mols = [(smi, chem_smiles.to_mol(smi)) for smi in smiles]
        return smiles_and_mols

    def _sample_smiles_and_calculate_loss(self, model, sample_size):
        sampled_smis, sampled_nlls = model.sample_smiles(num=sample_size)
        return sampled_smis, sampled_nlls

    def _calc_nlls(self, model, path, sample_size):
        return np.concatenate(
            list(md.calculate_nlls_from_model(model, chem_smiles.read_smiles_file(path, num=sample_size))[0]))

    def _update_nll_with_validation(self, sampled_nlls, validation_nlls, training_nlls):
        def jsd(dists):
            num_dists = len(dists)
            avg_dist = np.sum(dists, axis=0) / num_dists
            return np.sum([sps.entropy(dist, avg_dist) for dist in dists]) / num_dists

        self._data["jsd"] = {
            "sampled.validation": jsd([sampled_nlls, validation_nlls]),
            "sampled.training": jsd([sampled_nlls, training_nlls]),
            "training.validation": jsd([training_nlls, validation_nlls])
        }
        self._data["jsd_joined"] = jsd([sampled_nlls, training_nlls, validation_nlls])

    def _update_nll(self, sampled_nlls, training_nlls):
        def jsd(dists):
            num_dists = len(dists)
            avg_dist = np.sum(dists, axis=0) / num_dists
            return np.sum([sps.entropy(dist, avg_dist) for dist in dists]) / num_dists

        self._data["jsd"] = {"sampled.training": jsd([sampled_nlls, training_nlls])}
        self._data["jsd_joined"] = jsd([sampled_nlls, training_nlls])

    def _amplify_dataset(self, training_nlls: np.array, target_size: int):
        training_set_length = len(training_nlls)

        if training_set_length < target_size:
            delta = target_size - training_set_length
            padding = []
            counter = 0
            
            for i in range(delta):
                padding.append(training_nlls[counter])
                if training_set_length == (counter + 1):
                    counter = 0
                else:
                    counter += 1

            training_nlls = np.concatenate([training_nlls, padding])

        return training_nlls

    def log_out_inputs(self):
        self._logger.log_out_input_configuration()
