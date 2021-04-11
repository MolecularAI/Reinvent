import numpy as np
import requests
from reinvent_chemistry.conversions import Conversions

import running_modes.utils.configuration as utils_log
from reinvent_chemistry.logging import fraction_valid_smiles
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.sampling.logging.base_sampling_logger import BaseSamplingLogger


class RemoteSamplingLogger(BaseSamplingLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._is_dev = utils_log._is_development_environment()
        self._conversions = Conversions()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, smiles: [str], likelihoods: np.array):
        valid_smiles_fraction = fraction_valid_smiles(smiles)
        fraction_unique_entries = self._get_unique_entires_fraction(likelihoods)
        smiles_report = self._create_sample_report(smiles)
        data = self._assemble_timestep_report(valid_smiles_fraction, fraction_unique_entries, smiles_report)
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

    def _create_sample_report(self, smiles):
        legend, list_of_mols = self._count_unique_inchi_keys(smiles)
        list_of_smiles = [self._conversions.mol_to_smiles(mol) if mol is not None else "INVALID" for mol in list_of_mols]

        report = {
            "smiles": list_of_smiles,
            "legend": legend
        }
        return report

    def _assemble_timestep_report(self, fraction_valid_smiles, fraction_unique_smiles, smiles_report):
        timestep_report = {
            "fraction_unique_smiles": f"{fraction_unique_smiles} %",
            "fraction_valid_smiles": f"{fraction_valid_smiles} %",
            "smiles_report": smiles_report
        }
        return timestep_report
