import time
from typing import List

from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.loggable_data_dto import UpdateLoggableDataDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.actions import LinkInventSampleModel
from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.dto import SampledSequencesDTO, CurriculumOutcomeDTO, TimestepDTO, \
    UpdatedLikelihoodsDTO


class LinkInventCurriculumStrategy(BaseCurriculumStrategy):

    def run(self) -> CurriculumOutcomeDTO:
        step_counter = 0
        self.disable_prior_gradients()

        for item_id, sf_configuration in enumerate(self._parameters.curriculum_objectives):
            start_time = time.time()
            scoring_function = self._setup_scoring_function(item_id)
            step_counter = self.promote_agent(agent=self._agent, scoring_function=scoring_function,
                                              step_counter=step_counter, start_time=start_time,
                                              merging_threshold=sf_configuration.score_threshold)
            self.save_and_flush_memory(agent=self._agent, memory_name=f"_merge_{item_id}")
        is_successful_curriculum = step_counter < self._parameters.max_num_iterations
        outcome_dto = CurriculumOutcomeDTO(self._agent, step_counter, successful_curriculum=is_successful_curriculum)

        return outcome_dto

    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        # 1. Sampling
        sampled_sequences = self._sampling(agent)
        # 2. Scoring
        score_summary = self._scoring(scoring_function, sampled_sequences, step)
        # 3. Updating
        dto = self._updating(sampled_sequences, score_summary.total_score)
        # 4. Logging
        self._logging(start_time, step, score_summary, dto)

        score = score_summary.total_score.mean()
        return score

    def _sampling(self, agent) -> List[SampledSequencesDTO]:
        sampling_action = LinkInventSampleModel(agent, self._parameters.batch_size, self._logger,
                                                self._parameters.randomize_input)
        sampled_sequences = sampling_action.run(self._parameters.input)
        return sampled_sequences

    def _scoring(self, scoring_function,  sampled_sequences, step: int) -> FinalSummary:
        score_summary = self._apply_scoring_function(scoring_function, sampled_sequences, step)
        score_summary = self._clean_scored_smiles(score_summary)
        loggable_data = [UpdateLoggableDataDTO(dto.input, dto.output, dto.nll) for dto in sampled_sequences]
        dto = UpdateDiversityFilterDTO(score_summary, loggable_data, step)
        score_summary.total_score = self._diversity_filter.update_score(dto)
        return score_summary

    def _updating(self, sampled_sequences, score) -> UpdatedLikelihoodsDTO:
        likelihood_dto = self._agent.likelihood_smiles(sampled_sequences)
        dto = self.learning_strategy.run(likelihood_dto, score)
        return dto

    def _logging(self, start_time, step, score_summary, dto: UpdatedLikelihoodsDTO):
        report_dto = TimestepDTO(start_time, self._parameters.max_num_iterations, step, score_summary,
                                 dto.agent_likelihood, dto.prior_likelihood, dto.augmented_likelihood)
        self._logger.timestep_report(report_dto, self._diversity_filter, self._agent)

    def _apply_scoring_function(self, scoring_function, sampled_sequences: List[SampledSequencesDTO], step) -> FinalSummary:
        molecules = self._join_linker_and_warheads(sampled_sequences, keep_labels=True)
        smiles = []
        for idx, molecule in enumerate(molecules):
            try:
                smiles_str = self._conversion.mol_to_smiles(molecule) if molecule else "INVALID"
            except RuntimeError as exception:
                # NOTE: Current implementation of BondMaker (reinvent_chemistry.library_design.bond_maker) results in
                # impossible conversion of mol to smiles if one single atom has two attachment points and labels are
                # kept. As this case is not relevant in the context of link_invent, the can be discarded as invalid.
                smiles_str = "INVALID"
                self._logger.log_message(exception.__str__() + f'\n\tinput: {sampled_sequences[idx].input}'
                                        f'\n\toutput: {sampled_sequences[idx].output}\n')
            finally:
                smiles.append(smiles_str)
        final_score: FinalSummary = scoring_function.get_final_score_for_step(smiles, step)
        return final_score

    def _join_linker_and_warheads(self, sampled_sequences: List[SampledSequencesDTO], keep_labels=False):
        molecules = []
        for sample in sampled_sequences:
            linker = self._attachment_points.add_attachment_point_numbers(sample.output, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations(linker, sample.input,
                                                                       keep_labels_on_atoms=keep_labels)
            molecules.append(molecule)
        return molecules

    def _clean_scored_smiles(self, score_summary: FinalSummary) -> FinalSummary:
        """
        Remove attachment point numbers from scored smiles
        """
        # Note: method AttachmentPoints.remove_attachment_point_numbers does not work in this context, as it searches
        # for attachment point token ('*')
        score_summary.scored_smiles = [self._conversion.mol_to_smiles(
            self._attachment_points.remove_attachment_point_numbers_from_mol(self._conversion.smile_to_mol(smile))
            ) if idx in score_summary.valid_idxs else smile for idx, smile in enumerate(score_summary.scored_smiles)]
        return score_summary
