from typing import List

from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter

from running_modes.reinforcement_learning.configurations.scoring_strategy_configuration import ScoringStrategyConfiguration
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.reinforcement_learning.scoring_strategy.base_scoring_strategy import BaseScoringStrategy


class LinkInventScoringStrategy(BaseScoringStrategy):
    def __init__(self, strategy_configuration: ScoringStrategyConfiguration, diversity_filter: BaseDiversityFilter, logger):
        super().__init__(strategy_configuration, diversity_filter, logger)

    def evaluate(self, sampled_sequences: List[SampledSequencesDTO], step) -> FinalSummary:
        score_summary = self._apply_scoring_function(sampled_sequences)
        score_summary = self._clean_scored_smiles(score_summary)
        score_summary.total_score = self.diversity_filter.update_score(score_summary, sampled_sequences, step)
        return score_summary

    def _apply_scoring_function(self, sampled_sequences: List[SampledSequencesDTO]) -> FinalSummary:
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
                self.logger.log_message(exception.__str__() + f'\n\tinput: {sampled_sequences[idx].input}'
                                        f'\n\toutput: {sampled_sequences[idx].output}\n')
            finally:
                smiles.append(smiles_str)
        final_score: FinalSummary = self.scoring_function.get_final_score(smiles)
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
