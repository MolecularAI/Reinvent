from abc import ABC, abstractmethod
from typing import List

from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_scoring import FinalSummary, ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter

from running_modes.reinforcement_learning.configurations.scoring_strategy_configuration import \
    ScoringStrategyConfiguration
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO


class BaseScoringStrategy(ABC):
    def __init__(self, strategy_configuration: ScoringStrategyConfiguration, diversity_filter: BaseDiversityFilter, logger):
        self._configuration = strategy_configuration
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._conversion = Conversions()
        self.diversity_filter = diversity_filter
        self.scoring_function = ScoringFunctionFactory(strategy_configuration.scoring_function)
        self.logger = logger

    @abstractmethod
    def evaluate(self, sampled_sequences: List[SampledSequencesDTO], step: int) -> FinalSummary:
        raise NotImplemented("evaluate method is not implemented")

    def save_filter_memory(self):
        # TODO: might be good to consider separating the memory from the actual filter
        self.logger.save_filter_memory(self.diversity_filter)
