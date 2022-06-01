import os
import numpy as np
import requests
import torch
from reinvent_scoring import ScoringFunctionComponentNameEnum, FinalSummary, ComponentSpecificParametersEnum
from reinvent_scoring.scoring.diversity_filters.lib_invent.base_diversity_filter import BaseDiversityFilter

import running_modes.utils.configuration as ull
import reinvent_chemistry.logging as ul_rl
import running_modes.utils.general as ul_gen

from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLoggerConfiguration, \
    get_remote_logging_auth_token
from running_modes.reinforcement_learning.logging.link_logging.base_reinforcement_logger import BaseReinforcementLogger


class RemoteLinkReinforcementLogger(BaseReinforcementLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope, rl_config: ReinforcementLoggerConfiguration):
        super().__init__(configuration, rl_config)
        self._rows = 2
        self._columns = 5
        self._sample_size = self._rows * self._columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._specific_parameters_enum = ComponentSpecificParametersEnum()
        self._is_dev = ull._is_development_environment()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, start_time, n_steps, step, score_summary: FinalSummary,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, diversity_filter, actor):

        mean_score = np.mean(score_summary.total_score)
        smiles = score_summary.scored_smiles
        score = score_summary.total_score

        score_components = self._score_summary_breakdown(score_summary, mean_score)
        learning_curves = self._learning_curve_profile(agent_likelihood, prior_likelihood, augmented_likelihood)
        smiles_report = self._create_sample_report(smiles, score, score_summary)

        time_estimation = ul_gen.estimate_run_time(start_time, n_steps, step)
        data = self._assemble_timestep_report(step, score_components, diversity_filter, learning_curves,
                                              time_estimation, ul_rl.fraction_valid_smiles(smiles), smiles_report)
        self._notify_server(data, self._log_config.recipient)
        self.save_checkpoint(step, diversity_filter, actor)

    def save_final_state(self, agent, scaffold_filter):
        agent.save_to_file(os.path.join(self._log_config.result_folder, 'Agent.ckpt'))
        self.save_filter_memory(scaffold_filter)
        self.log_out_input_configuration()

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._logger.warning(f"posting to {to_address}")
            headers = {
                'Accept': 'application/json', 'Content-Type': 'application/json',
                'Authorization': get_remote_logging_auth_token()
            }
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
                smarts = summary_component.parameters.specific_parameters.get(self._specific_parameters_enum.SMILES, [])
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern

    def _visualize_structures(self, smiles, score, score_summary: FinalSummary):
        score, smiles = ul_rl.sort_smiles_by_score(score, smiles)
        smiles = ul_rl.padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = ul_rl.check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)
        pattern = ul_rl.find_matching_pattern_in_smiles(list_of_mols=list_of_mols, smarts_pattern=smarts_pattern)
        mol_in_base64_string = ul_rl.mol_to_png_string(list_of_mols, molsPerRow=self._columns, subImgSize=(300, 300),
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

    def _assemble_timestep_report(self, step, score_components, diversity_filter: BaseDiversityFilter,
                                  learning_curves, time_estimation, fraction_valid_smiles, smiles_report):
        actual_step = step + 1
        timestep_report = {"step": actual_step,
                           "components": score_components,
                           "learning": learning_curves,
                           "time_estimation": time_estimation,
                           "fraction_valid_smiles": fraction_valid_smiles,
                           "smiles_report": smiles_report,
                           "collected smiles in memory": diversity_filter.number_of_smiles_in_memory()
                           }
        return timestep_report
