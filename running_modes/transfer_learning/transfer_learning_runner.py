from typing import List

import rdkit
import torch
import torch.nn.utils as tnnu
import tqdm
from reinvent_chemistry.enums import FilterTypesEnum
from reinvent_chemistry.file_reader import FileReader
from reinvent_chemistry.standardization.filter_configuration import FilterConfiguration

import reinvent_models.reinvent_core.models.dataset as reinvent_dataset
import reinvent_models.reinvent_core.models.vocabulary as reinvent_vocabulary
from reinvent_models.reinvent_core.models.model import Model

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations.transfer_learning.transfer_learning_configuration import TransferLearningConfiguration
from running_modes.transfer_learning.adaptive_learning_rate import AdaptiveLearningRate
from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger

rdkit.rdBase.DisableLog("rdApp.error")


class TransferLearningRunner(BaseRunningMode):
    """Trains a given model."""

    def __init__(self, model: Model, config: TransferLearningConfiguration, logger: BaseTransferLearningLogger):
        self._model = model
        self._config = config
        self._logger = logger
        self._config.standardization_filters = self._set_standardization_filters(self._config.standardization_filters)
        self._reader = FileReader(self._config.standardization_filters, self._logger)
        adaptive_lr_config = self._config.adaptive_lr_config
        self._adaptive_learning_rate = AdaptiveLearningRate(model, self._logger, adaptive_lr_config, self._reader, self._config.standardize)

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
        training_set = self._reader.read_delimited_file(path, standardize=self._config.standardize,
                                                        randomize=self._config.randomize)
        dataset = reinvent_dataset.Dataset(smiles_list=training_set, vocabulary=self._model.vocabulary,
                                           tokenizer=reinvent_vocabulary.SMILESTokenizer())
        if len(dataset) == 0:
            raise IOError(f"No valid entries are present in the supplied file: {path}")

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

    def _set_standardization_filters(self, standardization_filters: List[dict]) -> List[FilterConfiguration]:
        if standardization_filters:
            filter_configs = [FilterConfiguration(**filters) for filters in standardization_filters]

            if self._config.validate_model_vocabulary:
                tokens = self._model.vocabulary.tokens()
                filter_types = FilterTypesEnum()
                config = FilterConfiguration(name=filter_types.VOCABULARY_FILTER, parameters={"vocabulary": tokens})
                filter_configs.append(config)

            return filter_configs
        else:
            return []
