import glob
import itertools as it
import os

from torch.utils.data import DataLoader
from reinvent_chemistry.file_reader import FileReader
from reinvent_models.link_invent.dataset.paired_dataset import PairedDataset
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

from running_modes.configurations.transfer_learning.link_invent_transfer_learning_configuration import \
    LinkInventTransferLearningConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.transfer_learning.link_invent_actions.collect_stats import CollectStats
from running_modes.transfer_learning.link_invent_actions.train_epoch import TrainEpoch
from running_modes.transfer_learning.logging.local_link_invent_transfer_learning_logger import \
    LocalLinkInventTransferLearningLogger


class LinkInventTransferLearningRunner(BaseRunningMode):
    def __init__(self, model: GenerativeModelBase, configuration: LinkInventTransferLearningConfiguration, optimizer,
                 learning_rate_scheduler, logger: LocalLinkInventTransferLearningLogger):
        self._model = model
        self._config = configuration
        self._optimizer = optimizer
        self._logger = logger
        self._lr_scheduler = learning_rate_scheduler

        self._reader = FileReader([], self._logger)
        self._training_data_sets = self._load_data_set(self._config.input_smiles_path)
        self._validation_data_set = self._load_data_set(self._config.validation_smiles_path) if \
            self._config.validation_smiles_path else None

        self._trained_model_path = os.path.join(self._config.output_path, 'trained_models')

    def run(self):
        self._set_up_output_folder()
        for epoch in self._get_epoch_range():

            self._logger.log_message(f'Working on epoch {epoch}')

            training_data = next(self._training_data_sets)
            validation_data = next(self._validation_data_set) if self._validation_data_set else None
            self._train_epoch(training_data)

            self._logging_stats(training_data=training_data, validation_data=validation_data, epoch=epoch,
                                learning_rate=self._lr_scheduler.optimizer.param_groups[0]["lr"])
            self._save_model_checkpoint(epoch=epoch)

            self._lr_scheduler.step()

            terminate = self._check_termination_criteria(epoch, self._lr_scheduler.optimizer.param_groups[0]["lr"])

            if terminate:
                self._model.save_to_file(os.path.join(self._config.output_path, self._config.model_file_name))
                break

    def _set_up_output_folder(self):
        os.makedirs(self._config.output_path, exist_ok=True)
        if self._config.save_model_frequency > 0:
            os.makedirs(self._trained_model_path, exist_ok=True)

    def _train_epoch(self, training_data):
        data_loader = self._initialize_data_loader(training_data, self._config.batch_size, drop_last=True)
        train_epoch_action = TrainEpoch(model=self._model, configuration=self._config, logger=self._logger,
                                        optimizer=self._optimizer, training_data_data_loader=data_loader,
                                        lr_scheduler=self._lr_scheduler)
        train_epoch_action.run()

    def _collect_stats(self, training_data, validation_data):
        stats_collector = CollectStats(model=self._model, training_data=training_data, validation_data=validation_data,
                                       logger=self._logger, sample_size=self._config.sample_size,
                                       initialize_data_loader_func=self._initialize_data_loader)
        stats = stats_collector.run()
        return stats

    def _check_termination_criteria(self, epoch, new_lr):
        terminate_flag = False
        self._lr_scheduler.step()
        if new_lr < self._config.learning_rate.min:
            self._logger.log_message("Reached LR minimum. Saving and terminating.")
            terminate_flag = True
        elif epoch == self._config.num_epochs:
            self._logger.log_message(f"Reached maximum number of epochs ({epoch}). Saving and terminating.")
            terminate_flag = True
        return terminate_flag

    def _logging_stats(self, training_data, validation_data, epoch: int, learning_rate: float):
        if self._config.collect_stats_frequency > 0 and epoch % self._config.collect_stats_frequency == 0:
            collected_stats = self._collect_stats(training_data, validation_data)
            self._logger.log_time_step(epoch=epoch, learning_rate=learning_rate, collected_stats=collected_stats,
                                       model=self._model)

    def _save_model_checkpoint(self, epoch):
        if self._config.save_model_frequency > 0 and epoch % self._config.save_model_frequency == 0:
            self._logger.log_message('Save model checkpoint')
            self._model.save_to_file(os.path.join(self._trained_model_path, f'model_{epoch:03d}.ckpt'))

    def _load_data_set(self, path_to_data_set: str):

        if os.path.isdir(path_to_data_set):
            file_paths = sorted(glob.glob(f"{path_to_data_set}/*.smi"))
        elif os.path.isfile(path_to_data_set):
            file_paths = [path_to_data_set]
        else:
            raise ValueError('path_to_data_set needs to be the path to a file or a folder')

        for path in it.cycle(file_paths):  # stores the path instead of the set
            dataset = list(self._reader.read_library_design_data_file(path, num_fields=2))

            if len(dataset) == 0:
                raise IOError(f"No valid entries are present in the supplied file: {path}")

            yield dataset

    def _get_epoch_range(self) -> range:
        last_epoch = self._config.starting_epoch + self._config.num_epochs - 1
        epoch_range = range(self._config.starting_epoch, last_epoch + 1)
        return epoch_range

    def _initialize_data_loader(self, data_set, batch_size, shuffle: bool = True, drop_last: bool = False):
        data_set = PairedDataset(input_target_smi_list=data_set, vocabulary=self._model.get_vocabulary())
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                 collate_fn=PairedDataset.collate_fn, drop_last=drop_last)
        return data_loader
