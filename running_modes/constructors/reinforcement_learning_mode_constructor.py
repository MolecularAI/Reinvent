from dacite import from_dict
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring import ScoringFunctionFactory

from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter import DiversityFilter
from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLearningComponents
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.reinforcement_runner import ReinforcementRunner
from running_modes.utils.general import set_default_device_cuda


class ReinforcementLearningModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        config = from_dict(data_class=ReinforcementLearningComponents, data=self._configuration.parameters)
        set_default_device_cuda()

        diversity_filter = DiversityFilter(config.diversity_filter)
        scoring_function = ScoringFunctionFactory(config.scoring_function)
        inception = Inception(config.inception, scoring_function, Model.load_from_file(config.reinforcement_learning.prior))
        runner = ReinforcementRunner(self._configuration, config.reinforcement_learning, diversity_filter, scoring_function, inception)
        return runner
