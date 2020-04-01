import rdkit
import torch
import torch.nn.utils as tnnu
import tqdm

import models.dataset as reinvent_dataset
import models.vocabulary as reinvent_vocabulary
import utils.smiles as chem_smiles
from models.model import Model
from running_modes.configurations.transfer_learning.transfer_learning_configuration import TransferLearningConfiguration
from running_modes.transfer_learning.adaptive_learning_rate import AdaptiveLearningRate

rdkit.rdBase.DisableLog("rdApp.error")


class TransferLearningRunner:
    """Trains a given model."""

    def __init__(self, model: Model, config: TransferLearningConfiguration, adaptive_learning_rate: AdaptiveLearningRate):
        self._model = model
        self._adaptive_learning_rate = adaptive_learning_rate
        self._config = config

    def run(self):
        last_epoch = self._config.starting_epoch + self._config.num_epochs - 1
        for epoch in range(self._config.starting_epoch, last_epoch + 1):
            if not self._adaptive_learning_rate.learning_rate_is_valid():
                break
            self._train_epoch(epoch, self._config.input_smiles_path)

        if self._config.save_every_n_epochs == 0 or (
                self._config.save_every_n_epochs != 1 and last_epoch % self._config.save_every_n_epochs > 0):
            self._save_model(last_epoch)
            self._adaptive_learning_rate.log_out_inputs()

    def _train_epoch(self, epoch, training_set_path):
        data_loader = self._initialize_dataloader(training_set_path)
        for _, batch in enumerate(self._progress_bar(data_loader, total=len(data_loader))):
            input_vectors = batch.long()
            loss = self._calculate_loss(input_vectors)

            self._adaptive_learning_rate.clear_gradient()
            loss.backward()
            if self._config.clip_gradient_norm > 0:
                tnnu.clip_grad_norm_(self._model.network.parameters(), self._config.clip_gradient_norm)
            self._adaptive_learning_rate.optimizer_step()

        if self._config.save_every_n_epochs > 0 and epoch % self._config.save_every_n_epochs == 0:
            model_path = self._save_model(epoch)
            self._calculate_stats_and_update_learning_rate(epoch, model_path)

    def _progress_bar(self, iterable, total, **kwargs):
        return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)

    def _initialize_dataloader(self, path):
        training_set = chem_smiles.read_smiles_file(path, standardize=self._config.standardize, randomize=self._config.randomize)
        dataset = reinvent_dataset.Dataset(smiles_list=training_set, vocabulary=self._model.vocabulary,
                                           tokenizer=reinvent_vocabulary.SMILESTokenizer())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self._config.batch_size,
                                                 shuffle=self._config.shuffle_each_epoch,
                                                 collate_fn=reinvent_dataset.Dataset.collate_fn)
        return dataloader

    def _calculate_loss(self, input_vectors):
        log_p = self._model.likelihood(input_vectors)
        return log_p.mean()

    def _save_model(self, epoch):
        self._model.save(self._model_path(epoch))
        return self._model_path(epoch)

    def _model_path(self, epoch):
        path = f"{self._config.output_model_path}.{epoch}" if epoch != self._config.num_epochs else f"{self._config.output_model_path}"
        return path

    def _calculate_stats_and_update_learning_rate(self, epoch, model_path):
        if self._config.collect_stats_frequency > 0 and epoch % self._config.collect_stats_frequency == 0:
            self._adaptive_learning_rate.collect_stats(epoch, model_path, self._config.input_smiles_path,
                                                       validation_set_path=self._config.validation_smiles_path)
        self._adaptive_learning_rate.update_lr_scheduler(epoch)
