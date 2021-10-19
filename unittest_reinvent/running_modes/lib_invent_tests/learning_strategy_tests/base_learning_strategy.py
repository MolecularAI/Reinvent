import os
import shutil
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from reinvent_chemistry.library_design.reaction_filters import ReactionFiltersEnum
from reinvent_chemistry.library_design.reaction_filters.reaction_filter_configruation import ReactionFilterConfiguration
from reinvent_scoring import ComponentParameters, ScoringFunctionComponentNameEnum, ScoringFunctionNameEnum
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.reinforcement_learning.configurations import LibInventReinforcementLearningConfiguration
from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.reinforcement_learning.configurations.lib_invent_scoring_strategy_configuration import \
    LibInventScoringStrategyConfiguration
from running_modes.reinforcement_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.reinforcement_learning.scoring_strategy.scoring_strategy_enum import ScoringStrategyEnum
from unittest_reinvent.fixtures.paths import LIBINVENT_PRIOR_PATH, MAIN_TEST_PATH
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, IBUPROFEN, SCAFFOLD_SUZUKI, CELECOXIB_SCAFFOLD, \
    DECORATION_SUZUKI


def dummy_func(a, b) -> float:
    return 0.3


class BaseTestLearningStrategy(unittest.TestCase):

    def setUp(self):
        self.sf_component_enum = ScoringFunctionComponentNameEnum()
        self.sf_enum = ScoringFunctionNameEnum()
        self.rf_enum = ReactionFiltersEnum()
        self.ss_enum = ScoringStrategyEnum()

        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)

        reactions = {
            "0": ["[*;$(c2aaaaa2),$(c2aaaa2):1]-!@[*;$(c2aaaaa2),$(c2aaaa2):2]>>[*:1][*].[*:2][*]"],
        }
        smiles = [CELECOXIB, ASPIRIN, IBUPROFEN]
        optimizer = Mock()
        optimizer.zero_grad = lambda: None
        optimizer.step = lambda: None

        critic_model = Mock()
        critic_model.set_mode = lambda _: None
        critic_model.network.parameters = lambda: []
        critic_model.likelihood = dummy_func


        reaction_filter_config = ReactionFilterConfiguration(type=self.rf_enum.SELECTIVE, reactions=reactions)
        diversity_filter_parameters = DiversityFilterParameters(name="NoFilter")
        tanimoto_similarity_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                                  specific_parameters={"smiles":smiles},
                                                                  component_type=self.sf_component_enum.TANIMOTO_SIMILARITY))
        scoring_function_parameters = ScoringFunctionParameters(name=self.sf_enum.CUSTOM_SUM,
                                                               parameters=[tanimoto_similarity_parameters])
        learning_strategy_config = LearningStrategyConfiguration(name=self.learning_strategy, parameters={"sigma": 100})
        scoring_strategy_configuration = LibInventScoringStrategyConfiguration(
            name=self.ss_enum.LIB_INVENT, scoring_function=scoring_function_parameters,
            diversity_filter=diversity_filter_parameters, reaction_filter=reaction_filter_config)

        config = LibInventReinforcementLearningConfiguration(actor=LIBINVENT_PRIOR_PATH, critic=LIBINVENT_PRIOR_PATH,
                                                    scaffolds=[SCAFFOLD_SUZUKI],
                                                    learning_strategy=learning_strategy_config,
                                                    scoring_strategy=scoring_strategy_configuration,
                                                             n_steps=2, batch_size=4)

        self.runner = LearningStrategy(critic_model, optimizer, config.learning_strategy)
        self.runner._to_tensor = self._to_tensor

        self.scaffold_batch = np.array([CELECOXIB_SCAFFOLD])
        self.decorator_batch = np.array([DECORATION_SUZUKI])
        self.score = torch.tensor([0.9], device=torch.device("cpu"))
        self.actor_nlls = torch.tensor([0.2], device=torch.device("cpu"), requires_grad=True)

    @staticmethod
    def _to_tensor(tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        return torch.autograd.Variable(tensor)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)


