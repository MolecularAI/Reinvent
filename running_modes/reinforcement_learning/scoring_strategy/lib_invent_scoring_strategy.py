from typing import List, Any

import numpy as np
from reinvent_chemistry.library_design.reaction_filters.reaction_filter import ReactionFilter
from reinvent_scoring import FinalSummary, ScoringFunctionComponentNameEnum, LoggableComponent, ComponentParameters, \
    ComponentSummary
from reinvent_scoring.scoring.diversity_filters.lib_invent.base_diversity_filter import BaseDiversityFilter

from running_modes.reinforcement_learning.configurations.lib_invent_scoring_strategy_configuration import \
    LibInventScoringStrategyConfiguration
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.reinforcement_learning.scoring_strategy.base_scoring_strategy import BaseScoringStrategy


class LibInventScoringStrategy(BaseScoringStrategy):
    def __init__(self, strategy_configuration: LibInventScoringStrategyConfiguration,
                 diversity_filter: BaseDiversityFilter, logger):
        super().__init__(strategy_configuration, diversity_filter, logger)
        self.reaction_filter = ReactionFilter(strategy_configuration.reaction_filter)

    def evaluate(self, sampled_sequences: List[SampledSequencesDTO], step) -> FinalSummary:
        score_summary = self._apply_scoring_function(sampled_sequences, step)

        score_summary.total_score = self.diversity_filter.update_score(score_summary, sampled_sequences, step)
        return score_summary

    def _apply_scoring_function(self, sampled_sequences: List[SampledSequencesDTO], step:int) -> FinalSummary:
        molecules = self._join_scaffolds_and_decorations(sampled_sequences)
        smiles = [self._conversion.mol_to_smiles(molecule) if molecule else "INVALID" for molecule in molecules]
        final_score: FinalSummary = self.scoring_function.get_final_score_for_step(smiles, step)
        final_score = self._apply_reaction_filters(molecules, final_score)
        return final_score

    def _apply_reaction_filters(self, molecules: List[Any], final_score: FinalSummary) -> FinalSummary:
        sf_component = ScoringFunctionComponentNameEnum()
        reaction_scores = [self.reaction_filter.evaluate(molecule) if molecule else 0 for molecule in molecules]
        loggable_component = LoggableComponent(name=sf_component.REACTION_FILTERS,
                                               component_type=sf_component.REACTION_FILTERS,
                                               score=reaction_scores)
        component_parameters = ComponentParameters(component_type=sf_component.REACTION_FILTERS,
                                                   name=sf_component.REACTION_FILTERS, weight=1)
        component_summary = ComponentSummary(total_score=reaction_scores, parameters=component_parameters)
        final_score.total_score = final_score.total_score * np.array(reaction_scores)
        final_score.scaffold_log.append(component_summary)
        final_score.profile.append(loggable_component)
        return final_score

    def _join_scaffolds_and_decorations(self, sampled_sequences: List[SampledSequencesDTO]):
        molecules = []
        for sample in sampled_sequences:
            scaffold = self._attachment_points.add_attachment_point_numbers(sample.input, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations(scaffold, sample.output)
            molecules.append(molecule)
        return molecules
