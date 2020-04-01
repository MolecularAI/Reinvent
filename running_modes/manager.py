import utils.general as utils_general
from models.model import Model
from running_modes.configurations import TransferLearningConfiguration, ScoringRunnerConfiguration, \
    ReinforcementLearningConfiguration, SampleFromModelConfiguration, CreateModelConfiguration, \
    AdaptiveLearningRateConfiguration, InceptionConfiguration, ReinforcementLearningComponents, \
    GeneralConfigurationEnvelope, ScoringRunnerComponents
from running_modes.create_model.create_model import CreateModelRunner
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.reinforcement_runner import ReinforcementRunner
from running_modes.sampling.sample_from_model import SampleFromModelRunner
from running_modes.scoring.scoring_runner import ScoringRunner
from running_modes.transfer_learning.adaptive_learning_rate import AdaptiveLearningRate
from running_modes.transfer_learning.transfer_learning_runner import TransferLearningRunner
from running_modes.validation.validation_runner import ValidationRunner
from scaffold.scaffold_filter_factory import ScaffoldFilterFactory
from scaffold.scaffold_parameters import ScaffoldParameters
from scoring.component_parameters import ComponentParameters
from scoring.scoring_function_factory import ScoringFunctionFactory
from scoring.scoring_function_parameters import ScoringFuncionParameters
from utils.enums.running_mode_enum import RunningModeEnum


class Manager:

    def __init__(self, configuration):
        self.running_mode_enum = RunningModeEnum()
        self.configuration = GeneralConfigurationEnvelope(**configuration)
        utils_general.set_default_device_cuda()

    def _run_create_empty_model(self):
        config = CreateModelConfiguration(**self.configuration.parameters)
        runner = CreateModelRunner(self.configuration, config)
        runner.run()

    def _run_transfer_learning(self):
        config = TransferLearningConfiguration(**self.configuration.parameters)
        model = Model.load_from_file(config.input_model_path)
        adaptive_lr_config = AdaptiveLearningRateConfiguration(**config.adaptive_lr_config)
        adaptive_learning_rate = AdaptiveLearningRate(model, self.configuration, adaptive_lr_config)
        runner = TransferLearningRunner(model, config, adaptive_learning_rate)
        runner.run()

    def _run_reinforcement_learning(self):
        rl_components = ReinforcementLearningComponents(**self.configuration.parameters)
        scaffold_filter = self._setup_scaffold_filter(rl_components.scaffold_filter)
        scoring_function = self._setup_scoring_function(rl_components.scoring_function)
        rl_config = ReinforcementLearningConfiguration(**rl_components.reinforcement_learning)
        inception_config = InceptionConfiguration(**rl_components.inception)
        inception = Inception(inception_config, scoring_function, Model.load_from_file(rl_config.prior))
        runner = ReinforcementRunner(self.configuration, rl_config, scaffold_filter, scoring_function,
                                     inception=inception)
        runner.run()

    def _setup_scaffold_filter(self, scaffold_parameters):
        scaffold_parameters = ScaffoldParameters(**scaffold_parameters)
        scaffold_factory = ScaffoldFilterFactory()
        scaffold = scaffold_factory.load_scaffold_filter(scaffold_parameters)
        return scaffold

    def _setup_scoring_function(self, scoring_function_parameters):
        scoring_function_parameters = ScoringFuncionParameters(**scoring_function_parameters)
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        return scoring_function_instance

    def _run_sampling(self):
        config = SampleFromModelConfiguration(**self.configuration.parameters)
        runner = SampleFromModelRunner(self.configuration, config)
        runner.run()

    def _run_scoring(self):
        sr_components = ScoringRunnerComponents(**self.configuration.parameters)
        scoring_function = self._setup_scoring_function(sr_components.scoring_function)
        scoring_config = ScoringRunnerConfiguration(**sr_components.scoring)
        runner = ScoringRunner(configuration=self.configuration,
                               config=scoring_config,
                               scoring_function=scoring_function)
        runner.run()

    def _run_validation(self):
        config = ComponentParameters(**self.configuration.parameters)
        runner = ValidationRunner(self.configuration, config)
        runner.run()

    def run(self):
        """determines from the configuration object which type of run it is expected to start"""
        switcher = {
            self.running_mode_enum.TRANSFER_LEARNING: self._run_transfer_learning,
            self.running_mode_enum.REINFORCEMENT_LEARNING: self._run_reinforcement_learning,
            self.running_mode_enum.SAMPLING: self._run_sampling,
            self.running_mode_enum.SCORING: self._run_scoring,
            self.running_mode_enum.CREATE_MODEL: self._run_create_empty_model,
            self.running_mode_enum.VALIDATION: self._run_validation
        }
        job = switcher.get(self.configuration.run_type, lambda: TypeError)
        job()
