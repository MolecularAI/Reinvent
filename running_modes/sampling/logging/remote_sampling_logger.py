import numpy as np
import requests

import utils.logging.log as utils_log
import utils as utils_general
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.sampling.logging.base_sampling_logger import BaseSamplingLogger
from utils.logging.visualization import mol_to_png_string


class RemoteSamplingLogger(BaseSamplingLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._is_dev = utils_log._is_development_environment()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, smiles: [str], likelihoods: np.array):
        fraction_valid_smiles = utils_general.fraction_valid_smiles(smiles)
        fraction_unique_entries = self._get_unique_entires_fraction(likelihoods)
        structures_table = self._visualize_structures(smiles)
        data = self._assemble_timestep_report(structures_table, fraction_valid_smiles, fraction_unique_entries)
        self._notify_server(data, self._log_config.recipient)

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._logger.warning(f"posting to {to_address}")
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            response = requests.post(to_address, json=data, headers=headers)

            if self._is_dev:
                """logs out the response content only when running a test instance"""
                if response.status_code == requests.codes.ok:
                    self._logger.info(f"SUCCESS: {response.status_code}")
                    self._logger.info(response.content)
                else:
                    self._logger.info(f"PROBLEM: {response.status_code}")
                    self._logger.exception(data, exc_info=False)
        except Exception as t_ex:
            self._logger.exception("Exception occurred", exc_info=True)
            self._logger.exception(f"Attempted posting the following data:")
            self._logger.exception(data, exc_info=False)

    def _assemble_timestep_report(self, structures_table, fraction_valid_smiles, fraction_unique_smiles):
        timestep_report = {
            "structures": structures_table,
            "fraction_unique_smiles": f"{fraction_unique_smiles} %",
            "fraction_valid_smiles": f"{fraction_valid_smiles} %"
        }
        return timestep_report

    def _visualize_structures(self, smiles):
        list_of_labels, list_of_mols = self._count_unique_inchi_keys(smiles)
        mol_in_base64_string = mol_to_png_string(list_of_mols, molsPerRow=self._columns, subImgSize=(300, 300),
                                                 legend=list_of_labels)
        return mol_in_base64_string
