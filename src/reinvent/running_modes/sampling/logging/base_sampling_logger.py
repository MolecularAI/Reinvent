import json
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
from rdkit.Chem import inchi
from rdkit import Chem
from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ...configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration


class BaseSamplingLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self._configuration = configuration
        self._log_config = SamplingLoggerConfiguration(**self._configuration.logging)
        self._setup_workfolder()
        self._logger = self._setup_logger()
        self._rows = 4
        self._columns = 5
        self._sample_size = self._rows * self._columns

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, smiles: [str], likelihoods: np.array):
        raise NotImplementedError("timestep_report method is not implemented")

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def _setup_workfolder(self):
        if not os.path.isdir(self._log_config.logging_path):
            os.makedirs(self._log_config.logging_path)

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("sampling_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def _count_unique_inchi_keys(self, smiles):
        """returns key value pair where value is [count, mol]"""
        inchi_dict = {}
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                inchi_key = inchi.MolToInchiKey(mol)
                try:
                    inchi_dict[inchi_key][0] += 1
                except:
                    inchi_dict[inchi_key] = [1, mol]
        counts = [v[0] for v in inchi_dict.values()]
        mols = [v[1] for v in inchi_dict.values()]
        to_sort = zip(counts, mols)
        sorted_tuple = sorted(to_sort, key=lambda tup: -tup[0])
        sorted_tuple = sorted_tuple[:self._sample_size]
        list_of_labels = [f"Times sampled: {v[0]}" for v in sorted_tuple]
        sorted_mols = [v[1] for v in sorted_tuple]
        return list_of_labels, sorted_mols

    def _get_unique_entires_fraction(self, some_list):
        return 100 * len(set(some_list)) / len(some_list)
