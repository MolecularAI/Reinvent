from abc import ABC, abstractmethod
from typing import List

from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.dto import CurriculumOutcomeDTO
from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.configurations.automated_curriculum_learning.prodcution_strategy_input_configuration import \
    ProductionStrategyInputConfiguration


class BaseProductionStrategy(ABC):
    def __init__(self, prior: GenerativeModelBase, diversity_filter: BaseDiversityFilter, inception: Inception,
                 scoring_function: BaseScoringFunction, configuration: ProductionStrategyInputConfiguration,
                 logger: BaseLogger):

        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._conversion = Conversions()
        self._parameters = configuration
        self._prior = prior
        self._logger = logger
        self._diversity_filter = diversity_filter
        self.inception = inception
        self._scoring_function = scoring_function

    @abstractmethod
    def run(self, cl_agent: GenerativeModelBase, steps_so_far: int) -> CurriculumOutcomeDTO:
        raise NotImplementedError("run() method is not implemented ")

    @abstractmethod
    def take_step(self, agent: GenerativeModelBase, learning_strategy, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        raise NotImplementedError("take_step() method is not implemented ")

    def disable_prior_gradients(self):
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def _log_sf_update(self, current_parameters: List[dict]):
        text_to_log = f"** Production setup **\n scoring_function: " \
                      f"{[component.get('name') for component in current_parameters]}"
        self._logger.log_message(text_to_log)

