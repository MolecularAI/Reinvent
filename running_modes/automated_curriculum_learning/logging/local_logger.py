import os

import numpy as np
from reinvent_chemistry.logging import padding_with_invalid_smiles, \
    check_for_invalid_mols_and_create_legend, find_matching_pattern_in_smiles, add_mols, fraction_valid_smiles
from reinvent_scoring.scoring.diversity_filters.lib_invent.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum, ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_summary import FinalSummary
from torch.utils.tensorboard import SummaryWriter

from running_modes.automated_curriculum_learning.dto.timestep_dto import TimestepDTO
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.automated_curriculum_learning.logging.console_message import ConsoleMessage
from running_modes.configurations import GeneralConfigurationEnvelope, CurriculumLoggerConfiguration


class LocalLogger(BaseLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope, log_config: CurriculumLoggerConfiguration):
        super().__init__(configuration, log_config)
        self._summary_writer = SummaryWriter(log_dir=self._log_config.logging_path)
        self._sample_size = self._log_config.rows * self._log_config.columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._specific_parameters_enum = ComponentSpecificParametersEnum()
        self._console_message_formatter = ConsoleMessage()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, report_dto: TimestepDTO, diversity_filter: BaseDiversityFilter, agent):
        message = self._console_message_formatter.create(report_dto)
        self._logger.info(message)
        self._tensorboard_report(report_dto, diversity_filter)
        self.save_checkpoint(report_dto.step, diversity_filter, agent)

    def save_final_state(self, agent, diversity_filter):
        agent.save_to_file(os.path.join(self._log_config.result_folder, 'Agent.ckpt'))
        self.save_filter_memory(diversity_filter)
        self._summary_writer.close()
        self.log_out_input_configuration()

    def _tensorboard_report(self, report_dto: TimestepDTO, diversity_filter: BaseDiversityFilter):
        self._summary_writer.add_scalars("nll/avg", {
            "prior": report_dto.prior_likelihood.mean(),
            "augmented": report_dto.augmented_likelihood.mean(),
            "agent": report_dto.agent_likelihood.mean()
        }, report_dto.step)
        self._summary_writer.add_scalars("nll/variance", {
            "prior": report_dto.prior_likelihood.var(),
            "augmented": report_dto.augmented_likelihood.var(),
            "agent": report_dto.agent_likelihood.var()
        }, report_dto.step)
        mean_score = np.mean(report_dto.score_summary.total_score)
        for i, log in enumerate(report_dto.score_summary.profile):
            self._summary_writer.add_scalar(report_dto.score_summary.profile[i].name,
                                            np.mean(report_dto.score_summary.profile[i].score), report_dto.step)
        self._summary_writer.add_scalar("Valid SMILES", fraction_valid_smiles(report_dto.score_summary.scored_smiles),
                                        report_dto.step)
        self._summary_writer.add_scalar("Number of SMILES found", diversity_filter.number_of_smiles_in_memory(),
                                        report_dto.step)
        self._summary_writer.add_scalar("Average score", mean_score, report_dto.step)
        self._log_out_smiles_sample(report_dto)

    def _log_out_smiles_sample(self, report_dto: TimestepDTO):
        self._visualize_structures(report_dto.score_summary.scored_smiles, report_dto.score_summary.total_score,
                                   report_dto.step, report_dto.score_summary)

    def _visualize_structures(self, smiles, score, step, score_summary: FinalSummary):
        list_of_mols, legends, pattern = self._check_for_invalid_mols_and_create_legends(smiles, score, score_summary)
        try:
            add_mols(self._summary_writer, "Molecules from epoch", list_of_mols[:self._sample_size], self._log_config.rows,
                     [x for x in legends], global_step=step, size_per_mol=(320, 320), pattern=pattern)
        except:
            self.log_message(f"Error in RDKit has occurred, skipping report for step {step}.")

    def _check_for_invalid_mols_and_create_legends(self, smiles, score, score_summary: FinalSummary):
        smiles = padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)
        pattern = find_matching_pattern_in_smiles(list_of_mols=list_of_mols, smarts_pattern=smarts_pattern)

        return list_of_mols, legend, pattern

    def _get_matching_substructure_from_config(self, score_summary: FinalSummary):
        smarts_pattern = ""
        for summary_component in score_summary.scaffold_log:
            if summary_component.parameters.component_type == self._sf_component_enum.MATCHING_SUBSTRUCTURE:
                smarts = summary_component.parameters.specific_parameters.get(self._specific_parameters_enum.SMILES, [])
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern
