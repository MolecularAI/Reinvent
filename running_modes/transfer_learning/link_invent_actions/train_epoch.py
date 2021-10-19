from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from torch.nn.utils import clip_grad_norm_

from running_modes.configurations.transfer_learning.link_invent_transfer_learning_configuration import \
    LinkInventTransferLearningConfiguration
from running_modes.transfer_learning.link_invent_actions.base_action import BaseAction
from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger


class TrainEpoch(BaseAction):
    def __init__(self, model: GenerativeModelBase, configuration: LinkInventTransferLearningConfiguration,
                 logger: BaseTransferLearningLogger, optimizer, training_data_data_loader, lr_scheduler):
        BaseAction.__init__(self, logger=logger)
        self._model = model
        self._config = configuration
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._data_loader = training_data_data_loader

        self._model_modes = ModelModeEnum()

    def run(self):
        self._logger.log_message(f'Train epoch')
        self._model.set_mode(self._model_modes.TRAINING)
        for input_batch, target_batch in self._data_loader:
            loss = self._model.likelihood(*input_batch, *target_batch).mean()
            self._optimizer.zero_grad()
            loss.backward()
            if self._config.clip_gradient_norm > 0:
                clip_grad_norm_(self._model.get_network_parameters(), self._config.clip_gradient_norm)
            self._optimizer.step()
