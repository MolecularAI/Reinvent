#!/usr/bin/env python
#  coding=utf-8

from ...models import model as reinvent
from ...models import vocabulary as voc
from ..configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration
from ..create_model.logging.create_model_logger import CreateModelLogger
from ..create_model.logging.remote_create_model_logger import RemoteCreateModelLogger
from ..configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ..configurations.create_model.create_model_configuration import CreateModelConfiguration
from ...utils import smiles as chem_smiles
from ...utils.enums.logging_mode_enum import LoggingModeEnum


class CreateModelRunner:

    def __init__(self, main_config: GeneralConfigurationEnvelope, configuration: CreateModelConfiguration):
        """
        Creates a CreateModelRunner.
        """
        self._smiles_list = chem_smiles.read_smiles_file(configuration.input_smiles_path, standardize=configuration.standardize)
        self._output_model_path = configuration.output_model_path

        self._num_layers = configuration.num_layers
        self._layer_size = configuration.layer_size
        self._cell_type = configuration.cell_type
        self._embedding_layer_size = configuration.embedding_layer_size
        self._dropout = configuration.dropout
        self._max_sequence_length = configuration.max_sequence_length
        self._layer_normalization = configuration.layer_normalization
        self.logger = self._resolve_logger(main_config)

    def run(self):
        """
        Carries out the creation of the model.
        """

        tokenizer = voc.SMILESTokenizer()
        vocabulary = voc.create_vocabulary(self._smiles_list, tokenizer=tokenizer)

        network_params = {
            'num_layers': self._num_layers,
            'layer_size': self._layer_size,
            'cell_type': self._cell_type,
            'embedding_layer_size': self._embedding_layer_size,
            'dropout': self._dropout,
            'layer_normalization': self._layer_normalization
        }
        model = reinvent.Model(no_cuda=True, vocabulary=vocabulary, tokenizer=tokenizer, network_params=network_params, max_sequence_length=self._max_sequence_length)
        model.save(self._output_model_path)
        return model

    def _resolve_logger(self, configuration: GeneralConfigurationEnvelope):
        logging_mode_enum = LoggingModeEnum()
        create_model_config = CreateModelLoggerConfiguration(**configuration.logging)
        if create_model_config.recipient == logging_mode_enum.LOCAL:
            logger = CreateModelLogger(configuration)
        else:
            logger = RemoteCreateModelLogger(configuration)
        return logger

