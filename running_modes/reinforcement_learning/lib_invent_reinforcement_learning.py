import time
from typing import List

from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.actions import LibInventSampleModel, LikelihoodEvaluation
from running_modes.reinforcement_learning.configurations.lib_invent_reinforcement_learning_configuration import \
    LibInventReinforcementLearningConfiguration
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.reinforcement_learning.learning_strategy import BaseLearningStrategy
from running_modes.reinforcement_learning.logging.link_logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.reinforcement_learning.scoring_strategy.base_scoring_strategy import BaseScoringStrategy


class LibInventReinforcementLearning(BaseRunningMode):

    def __init__(self, critic: GenerativeModelBase, actor: GenerativeModelBase, configuration: LibInventReinforcementLearningConfiguration,
                 learning_strategy: BaseLearningStrategy, scoring_strategy: BaseScoringStrategy, logger: BaseReinforcementLogger):
        self.critic = critic
        self.actor = actor
        self.configuration = self._double_single_scaffold_hack(configuration)
        self.logger = logger
        self.learning_strategy = learning_strategy
        self.scoring_strategy = scoring_strategy

    def _double_single_scaffold_hack(self, configuration: LibInventReinforcementLearningConfiguration) -> LibInventReinforcementLearningConfiguration:
        # trick to address the problem that the model requires a list of scaffolds.
        if len(configuration.scaffolds) == 1:
            configuration.scaffolds *= 2
            configuration.batch_size = max(int(configuration.batch_size/2), 1)
        return configuration

    def run(self):
        start_time = time.time()
        for step in range(self.configuration.n_steps):
            # 1. Sampling
            sampled_sequences = self._sampling()
            # 2. Scoring
            score_summary = self._scoring(sampled_sequences, step)
            # 3. Updating
            actor_nlls, critic_nlls, augmented_nlls = self._updating(sampled_sequences, score_summary.total_score)
            # 4. Logging
            self._logging(start_time, step, score_summary, actor_nlls, critic_nlls, augmented_nlls)
        self.scoring_strategy.save_filter_memory()

    def _sampling(self) -> List[SampledSequencesDTO]:
        sampling_action = LibInventSampleModel(self.actor, self.configuration.batch_size, self.logger,
                                               self.configuration.randomize_scaffolds)
        sampled_sequences = sampling_action.run(self.configuration.scaffolds)
        return sampled_sequences

    def _scoring(self, sampled_sequences, step: int) -> FinalSummary:
        return self.scoring_strategy.evaluate(sampled_sequences, step)

    def _updating(self, sampled_sequences, score):
        scaffold_batch, decorator_batch, actor_nlls = self._calculate_likelihood(sampled_sequences)
        actor_nlls, critic_nlls, augmented_nlls = self.learning_strategy.run(scaffold_batch, decorator_batch, score, actor_nlls)
        return actor_nlls, critic_nlls, augmented_nlls

    def _logging(self, start_time, step, score_summary, actor_nlls, critic_nlls, augmented_nlls):
        self.logger.timestep_report(start_time, self.configuration.n_steps, step, score_summary, actor_nlls,
                                    critic_nlls, augmented_nlls, self.scoring_strategy.diversity_filter, self.actor)

    def _calculate_likelihood(self, sampled_sequences: List[SampledSequencesDTO]):
        nll_calculation_action = LikelihoodEvaluation(self.actor, self.configuration.batch_size, self.logger)
        encoded_scaffold, encoded_decorators, nlls = nll_calculation_action.run(sampled_sequences)
        return encoded_scaffold, encoded_decorators, nlls
