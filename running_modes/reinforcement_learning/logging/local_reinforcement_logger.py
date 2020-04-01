import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils.logging.reinforcement_learning as ul_rl
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.reinforcement_learning.logging import ConsoleMessage
from running_modes.reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from scoring.score_summary import FinalSummary
from utils import fraction_valid_smiles
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.logging.tensorboard import add_mols


class LocalReinforcementLogger(BaseReinforcementLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._summary_writer = SummaryWriter(log_dir=self._log_config.logging_path)
        self._summary_writer.add_text('Legends',
                                      'The values under each compound are read as: [Agent; Prior; Target; Score]')
        # _rows and _columns define the shape of the output grid of molecule images in tensorboard.
        self._rows = 4
        self._columns = 4
        self._sample_size = self._rows * self._columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._console_message_formatter = ConsoleMessage()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, start_time, n_steps, step, smiles: np.array,
                        mean_score: np.float32, score_summary: FinalSummary, score: np.array,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor):
        message = self._console_message_formatter.create(start_time, n_steps, step, smiles, mean_score, score_summary,
                                                         score, agent_likelihood, prior_likelihood,
                                                         augmented_likelihood)
        self._logger.info(message)
        self._tensorboard_report(step, smiles, score, score_summary, agent_likelihood, prior_likelihood,
                                 augmented_likelihood)

    def save_final_state(self, agent, scaffold_filter):
        agent.save(os.path.join(self._log_config.resultdir, 'Agent.ckpt'))
        scaffold_filter.save_to_csv(self._log_config.resultdir, self._log_config.job_name)
        self._summary_writer.close()
        self.log_out_input_configuration()

    def _tensorboard_report(self, step, smiles, score, score_summary: FinalSummary, agent_likelihood, prior_likelihood,
                            augmented_likelihood):
        self._summary_writer.add_scalars("nll/avg", {
            "prior": prior_likelihood.mean(),
            "augmented": augmented_likelihood.mean(),
            "agent": agent_likelihood.mean()
        }, step)
        mean_score = np.mean(score)
        for i, log in enumerate(score_summary.profile):
            self._summary_writer.add_scalar(score_summary.profile[i].name, np.mean(score_summary.profile[i].score),
                                            step)
        self._summary_writer.add_scalar("average score", mean_score, step)
        self._summary_writer.add_scalar("Fraction valid SMILES", fraction_valid_smiles(smiles), step)
        if step % 10 == 0:
            self._log_out_smiles_sample(smiles, score, step, score_summary)

    def _log_out_smiles_sample(self, smiles, score, step, score_summary: FinalSummary):
        self._visualize_structures(smiles, score, step, score_summary)

    def _visualize_structures(self, smiles, score, step, score_summary: FinalSummary):

        list_of_mols, legends, pattern = self._check_for_invalid_mols_and_create_legends(smiles, score, score_summary)
        try:
            add_mols(self._summary_writer, "Molecules from epoch", list_of_mols[:self._sample_size], self._rows,
                     [x for x in legends], global_step=step, size_per_mol=(320, 320), pattern=pattern)
        except Exception as ex:
            print(f"Error in RDKit has occurred, skipping printout for step {step}.")

    def _check_for_invalid_mols_and_create_legends(self, smiles, score, score_summary: FinalSummary):
        smiles = ul_rl.padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = ul_rl.check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)
        pattern = ul_rl.find_matching_pattern_in_smiles(list_of_mols=list_of_mols, smarts_pattern=smarts_pattern)

        return list_of_mols, legend, pattern

    def _get_matching_substructure_from_config(self, score_summary: FinalSummary):
        smarts_pattern = ""
        for summary_component in score_summary.scaffold_log:
            if summary_component.parameters.component_type == self._sf_component_enum.MATCHING_SUBSTRUCTURE:
                smarts = summary_component.parameters.smiles
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern
