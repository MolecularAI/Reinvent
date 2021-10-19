import os

from reinvent_chemistry.file_reader import FileReader
from reinvent_models.link_invent.link_invent_model import LinkInventModel
from reinvent_models.link_invent.model_vocabulary.paired_model_vocabulary import PairedModelVocabulary
from reinvent_models.link_invent.networks import EncoderDecoder

from running_modes.configurations import LinkInventCreateModelConfiguration
from running_modes.create_model.logging.base_create_model_logger import BaseCreateModelLogger
from reinvent_models.model_factory.enums.model_parameter_enum import ModelParametersEnum


class LinkInventCreateModelRunner:
    def __init__(self, configuration: LinkInventCreateModelConfiguration, logger: BaseCreateModelLogger):
        self._configuration = configuration
        self._logger = logger
        self._reader = FileReader([], None)

    def run(self):
        vocabulary = self._build_vocabulary()
        model = self._get_model(vocabulary)

        self._save_model(model)
        return model

    def _build_vocabulary(self) -> PairedModelVocabulary:
        self._logger.log_message('Building vocabulary')

        warheads_list, linker_list = zip(
            *self._reader.read_library_design_data_file(self._configuration.input_smiles_path, num_fields=2))

        model_vocabulary = PairedModelVocabulary.from_lists(warheads_list, linker_list)
        self._logger.log_message("Warheads vocabulary contains {} tokens: {}".format(
            len(model_vocabulary.input), model_vocabulary.input.vocabulary.tokens()))
        self._logger.log_message("Linker vocabulary contains {} tokens: {}".format(
            len(model_vocabulary.target), model_vocabulary.target.vocabulary.tokens()))

        return model_vocabulary

    def _get_model(self, model_vocabulary: PairedModelVocabulary):
        parameter_enum = ModelParametersEnum()
        encoder_config = {
            parameter_enum.NUMBER_OF_LAYERS: self._configuration.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self._configuration.layer_size,
            parameter_enum.DROPOUT: self._configuration.dropout,
            parameter_enum.VOCABULARY_SIZE: len(model_vocabulary.input)
        }
        decoder_config = {
            parameter_enum.NUMBER_OF_LAYERS: self._configuration.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self._configuration.layer_size,
            parameter_enum.DROPOUT: self._configuration.dropout,
            parameter_enum.VOCABULARY_SIZE: len(model_vocabulary.target)
        }

        network = EncoderDecoder(encoder_config, decoder_config)
        model = LinkInventModel(vocabulary=model_vocabulary, network=network,
                                max_sequence_length=self._configuration.max_sequence_length)
        return model

    def _save_model(self, model: LinkInventModel):
        self._logger.log_message(f'Saving model at {self._configuration.output_model_path}')
        os.makedirs(os.path.dirname(self._configuration.output_model_path), exist_ok=True)
        model.save_to_file(self._configuration.output_model_path)
        self._logger.log_out_input_configuration()
