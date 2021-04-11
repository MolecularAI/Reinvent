import numpy as np
import requests
from reinvent_chemistry.conversions import Conversions

import running_modes.utils.configuration as utils_log
from reinvent_chemistry.logging import fraction_valid_smiles
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger


class RemoteTransferLearningLogger(BaseTransferLearningLogger):

    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._is_dev = utils_log._is_development_environment()
        self._conversions = Conversions()

    def log_message(self, message: str):
        self._logger.info(message)

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

    def log_timestep(self, lr, epoch, sampled_smiles, sampled_nlls,
                     validation_nlls, training_nlls, jsd_data, jsd_joined_data, model, model_path):

        learning_mean = self._mean_learning_curve_profile(sampled_nlls, training_nlls)
        learning_variation = self._variation_learning_curve_profile(sampled_nlls, training_nlls)
        valid_smiles_fraction = fraction_valid_smiles(sampled_smiles)
        smiles_report = self._create_sample_report(sampled_smiles)

        data = self._assemble_timestep_report(epoch, valid_smiles_fraction, learning_mean,
                                              learning_variation, sampled_nlls, training_nlls, smiles_report,
                                              model_path)
        self._notify_server(data, self._log_config.recipient)

    def _create_sample_report(self, smiles):
        legend, list_of_mols = self._count_compound_frequency(smiles)
        list_of_smiles = [self._conversions.mol_to_smiles(mol) if mol is not None else "INVALID" for mol in list_of_mols]

        report = {
            "smiles": list_of_smiles,
            "legend": legend
        }
        return report

    def _mean_learning_curve_profile(self, sampled_nlls: np.array, training_nlls: np.array):
        learning_curves = {
            "sampled": float(np.float(sampled_nlls.mean())),
            "training": float(np.float(training_nlls.mean()))
        }
        return learning_curves

    def _variation_learning_curve_profile(self, sampled_nlls: np.array, training_nlls: np.array):
        learning_curves = {
            "sampled": float(np.float(sampled_nlls.var())),
            "training": float(np.float(training_nlls.var()))
        }
        return learning_curves

    def _assemble_timestep_report(self, epoch, fraction_valid_smiles, learning_mean,
                                  learning_variation, sampled_nlls, training_nlls, smiles_report,
                                  model_path) -> dict:
        timestep_report = {"epoch": epoch,
                           "fraction_valid_smiles": {"valid": fraction_valid_smiles},
                           "sampled_smiles_distribution": {"negative_log_likelihood": sampled_nlls.tolist()},
                           "training_smiles_distribution": {"negative_log_likelihood": training_nlls.tolist()},
                           "learning_mean": learning_mean,
                           "learning_variation": learning_variation,
                           "smiles_report": smiles_report,
                           "model_path": model_path
                           }
        return timestep_report
