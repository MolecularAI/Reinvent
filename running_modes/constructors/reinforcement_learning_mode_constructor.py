import reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter as lib_invent_df
import torch
from dacite import from_dict
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_scoring.scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter

from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLearningComponents, \
    ReinforcementLoggerConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.reinforcement_learning import LibInventReinforcementLearning, LinkInventReinforcementLearning, \
    CoreReinforcementRunner, Inception
from running_modes.reinforcement_learning.configurations import LinkInventReinforcementLearningConfiguration, \
    LibInventReinforcementLearningConfiguration
from running_modes.reinforcement_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.reinforcement_learning.scoring_strategy.scoring_strategy import ScoringStrategy
from running_modes.utils.general import set_default_device_cuda


class ReinforcementLearningModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        set_default_device_cuda()
        model_type_enum = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()

        if self._configuration.model_type == model_type_enum.DEFAULT:

            log_config = ReinforcementLoggerConfiguration.parse_obj(self._configuration.logging)
            logger = ReinforcementLogger(self._configuration, log_config)
            config = from_dict(data_class=ReinforcementLearningComponents, data=self._configuration.parameters)

            prior_config = ModelConfiguration(self._configuration.model_type, model_regime.INFERENCE, config.reinforcement_learning.prior)
            actor_config = ModelConfiguration(self._configuration.model_type, model_regime.TRAINING, config.reinforcement_learning.agent)
            prior = GenerativeModel(prior_config)
            actor = GenerativeModel(actor_config)
            assert prior.get_vocabulary() == actor.get_vocabulary(), "The agent and the prior must have the same vocabulary"

            diversity_filter = DiversityFilter(config.diversity_filter)
            scoring_function = ScoringFunctionFactory(config.scoring_function)
            inception = Inception(config.inception, scoring_function, prior)
            runner = CoreReinforcementRunner(prior, actor, config.reinforcement_learning,
                                             scoring_function, diversity_filter, inception, logger)

        elif self._configuration.model_type == model_type_enum.LIB_INVENT:

            config=from_dict(data_class=LibInventReinforcementLearningConfiguration,data=self._configuration.parameters)
            logging_config = ReinforcementLoggerConfiguration.parse_obj(self._configuration.logging)
            logger = ReinforcementLogger(self._configuration, logging_config)

            critic_config = ModelConfiguration(self._configuration.model_type, model_regime.INFERENCE, config.critic)
            actor_config = ModelConfiguration(self._configuration.model_type, model_regime.TRAINING, config.actor)
            critic = GenerativeModel(critic_config)
            actor = GenerativeModel(actor_config)

            optimizer = torch.optim.Adam(actor.get_network_parameters(), lr=config.learning_rate)
            learning_strategy = LearningStrategy(critic, optimizer, config.learning_strategy, logger)
            diversity_filter = lib_invent_df.DiversityFilter(config.scoring_strategy.diversity_filter)
            scoring_strategy = ScoringStrategy(config.scoring_strategy, diversity_filter, logger)
            runner = LibInventReinforcementLearning(critic=critic, actor=actor, configuration=config,
                                                    learning_strategy=learning_strategy,
                                                    scoring_strategy=scoring_strategy, logger=logger)

        elif self._configuration.model_type == model_type_enum.LINK_INVENT:

            config = from_dict(data_class=LinkInventReinforcementLearningConfiguration,
                               data=self._configuration.parameters)
            logging_config = ReinforcementLoggerConfiguration.parse_obj(self._configuration.logging)
            logger = ReinforcementLogger(self._configuration, logging_config)
            critic_config = ModelConfiguration(self._configuration.model_type, model_regime.INFERENCE, config.critic)
            actor_config = ModelConfiguration(self._configuration.model_type, model_regime.TRAINING, config.actor)
            critic = GenerativeModel(critic_config)
            actor = GenerativeModel(actor_config)

            optimizer = torch.optim.Adam(actor.get_network_parameters(), lr=config.learning_rate)
            learning_strategy = LearningStrategy(critic, optimizer, config.learning_strategy, logger)

            diversity_filter = lib_invent_df.DiversityFilter(config.scoring_strategy.diversity_filter)
            scoring_strategy = ScoringStrategy(config.scoring_strategy, diversity_filter, logger)
            runner = LinkInventReinforcementLearning(critic=critic, actor=actor, configuration=config,
                                                     learning_strategy=learning_strategy,
                                                     scoring_strategy=scoring_strategy, logger=logger)
        else:
            raise KeyError(f"Incorrect model type: `{self._configuration.model_type}` provided")
        return runner
