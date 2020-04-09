import os

import numpy as np
import requests
import torch

from ....utils.logging import log as ull
from ....utils.logging import reinforcement_learning as ul_rl
from ...configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from ...reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from ....scoring.score_summary import FinalSummary
from ....utils import fraction_valid_smiles
from ....utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from ....utils.logging.visualization import mol_to_png_string


class RemoteReinforcementLogger(BaseReinforcementLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._rows = 2
        self._columns = 5
        self._sample_size = self._rows * self._columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._is_dev = ull._is_development_environment()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, start_time, n_steps, step, smiles,
                        mean_score: np.array, score_summary: FinalSummary, score,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor):
        score_components = self._score_summary_breakdown(score_summary, mean_score)
        learning_curves = self._learning_curve_profile(agent_likelihood, prior_likelihood, augmented_likelihood)
        structures_table = self._visualize_structures(smiles, score, score_summary)
        smiles_report = self._create_sample_report(smiles, score, score_summary)

        time_estimation = ul_rl.estimate_run_time(start_time, n_steps, step)
        data = self._assemble_timestep_report(step, score_components, structures_table, learning_curves,
                                              time_estimation, fraction_valid_smiles(smiles), smiles_report)
        self._notify_server(data, self._log_config.recipient)

    def save_final_state(self, agent, scaffold_filter):
        agent.save(os.path.join(self._log_config.resultdir, 'Agent.ckpt'))
        scaffold_filter.save_to_csv(self._log_config.resultdir, self._log_config.job_name)
        self.log_out_input_configuration()

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

    def _get_matching_substructure_from_config(self, score_summary: FinalSummary):
        smarts_pattern = ""
        for summary_component in score_summary.scaffold_log:
            if summary_component.parameters.component_type == self._sf_component_enum.MATCHING_SUBSTRUCTURE:
                smarts = summary_component.parameters.smiles
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern

    def _visualize_structures(self, smiles, score, score_summary: FinalSummary):
        score, smiles = ul_rl.sort_smiles_by_score(score, smiles)
        smiles = ul_rl.padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = ul_rl.check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)
        pattern = ul_rl.find_matching_pattern_in_smiles(list_of_mols=list_of_mols, smarts_pattern=smarts_pattern)
        mol_in_base64_string = mol_to_png_string(list_of_mols, molsPerRow=self._columns, subImgSize=(300, 300),
                                                 legend=legend, matches=pattern)
        return mol_in_base64_string

    def _create_sample_report(self, smiles, score, score_summary: FinalSummary):
        score, smiles = ul_rl.sort_smiles_by_score(score, smiles)
        smiles = ul_rl.padding_with_invalid_smiles(smiles, self._sample_size)
        _, legend = ul_rl.check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)

        smiles_legend_pairs = [{"smiles": smiles[indx], "legend": legend[indx]} for indx in range(self._sample_size)]

        report = {
            "smarts_pattern": smarts_pattern,
            "smiles_legend_pairs": smiles_legend_pairs
        }
        return report

    def _learning_curve_profile(self, agent_likelihood, prior_likelihood, augmented_likelihood):
        learning_curves = {
            "prior": float(np.float(prior_likelihood.detach().mean().cpu())),
            "augmented": float(np.float(augmented_likelihood.detach().mean().cpu())),
            "agent": float(np.float(agent_likelihood.detach().mean().cpu()))
        }
        return learning_curves

    def _score_summary_breakdown(self, score_summary: FinalSummary, mean_score: np.array):
        score_components = {}
        for i, log in enumerate(score_summary.profile):
            score_components[f"{score_summary.profile[i].component_type}:{score_summary.profile[i].name}"] = \
                float(np.mean(score_summary.profile[i].score))
        score_components["total_score:total_score"] = float(mean_score)
        return score_components

    def _assemble_timestep_report(self, step, score_components, structures_table, learning_curves, time_estimation,
                                  fraction_valid_smiles, smiles_report):
        actual_step = step + 1
        timestep_report = {"step": actual_step,
                           "components": score_components,
                           "structures": structures_table,
                           "learning": learning_curves,
                           "time_estimation": time_estimation,
                           "fraction_valid_smiles": fraction_valid_smiles,
                           "smiles_report": smiles_report
                           }
        return timestep_report
