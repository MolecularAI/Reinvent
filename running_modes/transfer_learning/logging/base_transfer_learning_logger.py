import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Tuple, List

from rdkit import Chem
from rdkit.Chem import inchi

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.transfer_learning_log_configuration import TransferLearningLoggerConfig


class BaseTransferLearningLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self._configuration = configuration
        self._log_config = TransferLearningLoggerConfig(**self._configuration.logging)
        self._logger = self._setup_logger()
        self._with_weights = self._log_config.use_weights
        self._rows = 4
        self._columns = 5
        self._sample_size = self._rows * self._columns

    def log_message(self, message: str):
        self._logger.info(message)

    @abstractmethod
    def log_timestep(self, lr, epoch, sampled_smiles, sampled_nlls,
                     validation_nlls, training_nlls, jsd_data, jsd_joined_data, model):
        raise NotImplementedError("log_timestep method is not implemented")

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def _setup_logger(self):
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("transfer_learning_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def _count_unique_inchi_keys(self, smiles) -> Tuple[List, List]:
        """returns key value pair where value is [count, mol]"""
        inchi_dict = {}
        for smile in smiles:
            self._append_inchi_keys_dictionary_by_reference(inchi_dict, smile)
        counts = [v[0] for v in inchi_dict.values()]
        mols = [v[1] for v in inchi_dict.values()]
        to_sort = zip(counts, mols)
        sorted_tuple = sorted(to_sort, key=lambda tup: -tup[0])
        sorted_tuple = sorted_tuple[:self._sample_size]
        list_of_labels = [f"Times sampled: {v[0]}" for v in sorted_tuple]
        sorted_mols = [v[1] for v in sorted_tuple]
        return list_of_labels, sorted_mols

    def _append_inchi_keys_dictionary_by_reference(self, inchi_dict: dict, smile: str):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            try:
                inchi_key = inchi.MolToInchiKey(mol)
                try:
                    inchi_dict[inchi_key][0] += 1
                except:
                    inchi_dict[inchi_key] = [1, mol]
            except:
                self.log_message(f"Failed to transform SMILES string: {smile}")
