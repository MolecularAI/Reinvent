import os
import shutil
import unittest

from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_scoring import ScoringFunctionParameters, ScoringFunctionNameEnum, ComponentParameters, \
    ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters
from reinvent_scoring.scoring.enums.diversity_filter_enum import DiversityFilterEnum
from torch.optim import Adam

from running_modes.configurations import ReinforcementLoggerConfiguration, GeneralConfigurationEnvelope
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.reinforcement_learning.configurations import LinkInventReinforcementLearningConfiguration
from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.reinforcement_learning.configurations.link_invent_scoring_strategy_congfiguration import \
    LinkInventScoringStrategyConfiguration
from running_modes.reinforcement_learning.learning_strategy import LearningStrategyEnum
from running_modes.reinforcement_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.reinforcement_learning.link_invent_reinforcement_learning import LinkInventReinforcementLearning
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.reinforcement_learning.scoring_strategy.link_invent_scoring_strategy import LinkInventScoringStrategy
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, LINK_INVENT_PRIOR_PATH
from unittest_reinvent.fixtures.test_data import WARHEAD_PAIR


class TestLinkInventReinforcementLearningRunner(unittest.TestCase):
    def setUp(self) -> None:

        set_default_device_cuda()

        mt_enum = ModelTypeEnum()
        rt_enum = RunningModeEnum()
        lt_enum = LoggingModeEnum()
        mr_enum = GenerativeModelRegimeEnum()
        ls_enum = LearningStrategyEnum()
        sf_enum = ScoringFunctionNameEnum()
        sfcn_enum = ScoringFunctionComponentNameEnum()
        df_enum = DiversityFilterEnum()

        self.work_dir = os.path.join(MAIN_TEST_PATH, mt_enum.LINK_INVENT+rt_enum.REINFORCEMENT_LEARNING)
        os.makedirs(self.work_dir, exist_ok=True)

        self.log_dir = os.path.join(self.work_dir, 'log_dir')
        logging_config = ReinforcementLoggerConfiguration(logging_path=self.log_dir, recipient=lt_enum.LOCAL, result_folder=self.log_dir)
        learning_strategy_config = LearningStrategyConfiguration(name=ls_enum.DAP, parameters=dict(sigma=120))
        component_parameter_config = ComponentParameters(component_type=sfcn_enum.QED_SCORE, name=sfcn_enum.QED_SCORE,
                                                         weight=1)
        scoring_function_parameters = ScoringFunctionParameters(name=sf_enum.CUSTOM_SUM,
                                                                parameters=[component_parameter_config.__dict__])
        diversity_filter_parameters = DiversityFilterParameters(name=df_enum.NO_FILTER)
        scoring_strategy_configuration = LinkInventScoringStrategyConfiguration(
            name=mt_enum.LINK_INVENT, scoring_function=scoring_function_parameters,
            diversity_filter=diversity_filter_parameters)

        config_envelope = GeneralConfigurationEnvelope({}, {}, rt_enum.REINFORCEMENT_LEARNING, "3", model_type= mt_enum.LINK_INVENT)
        logger = ReinforcementLogger(config_envelope, logging_config)

        critic = GenerativeModel(ModelConfiguration(model_type=mt_enum.LINK_INVENT, model_mode=mr_enum.INFERENCE,
                                                    model_file_path=LINK_INVENT_PRIOR_PATH))
        actor = GenerativeModel(ModelConfiguration(model_type=mt_enum.LINK_INVENT, model_mode=mr_enum.TRAINING,
                                                   model_file_path=LINK_INVENT_PRIOR_PATH))
        optimizer = Adam(actor.get_network_parameters(), lr=0.0001)
        learning_strategy = LearningStrategy(critic, optimizer, learning_strategy_config, logger)
        diversity_filter = DiversityFilter(scoring_strategy_configuration.diversity_filter)
        scoring_strategy = LinkInventScoringStrategy(scoring_strategy_configuration, diversity_filter, logger)
        config = LinkInventReinforcementLearningConfiguration(
            actor=LINK_INVENT_PRIOR_PATH, critic=LINK_INVENT_PRIOR_PATH, warheads=[WARHEAD_PAIR],
            learning_strategy=learning_strategy_config, scoring_strategy=scoring_strategy_configuration,
            randomize_warheads=False, n_steps=2)
        self.runner = LinkInventReinforcementLearning(critic=critic, actor=actor, configuration=config,
                                                      learning_strategy=learning_strategy,
                                                      scoring_strategy=scoring_strategy, logger=logger)

    def tearDown(self) -> None:
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)

    def test_runner(self):
        self.runner.run()
        self.assertTrue(os.path.isdir(self.log_dir))
        self.assertTrue([f for f in os.listdir(self.log_dir) if f.endswith('.csv')])  # memory csv file exists

