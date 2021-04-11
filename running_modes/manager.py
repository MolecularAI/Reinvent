import json
import os

import running_modes.utils.general as utils_general
from models.model import Model
from running_modes.configurations import TransferLearningConfiguration, ScoringRunnerConfiguration, \
    ReinforcementLearningConfiguration, SampleFromModelConfiguration, CreateModelConfiguration, \
    InceptionConfiguration, ReinforcementLearningComponents, \
    GeneralConfigurationEnvelope, ScoringRunnerComponents
from running_modes.create_model.create_model import CreateModelRunner
from running_modes.curriculum_learning.curriculum_runner import CurriculumRunner
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.reinforcement_runner import ReinforcementRunner
from running_modes.sampling.sample_from_model import SampleFromModelRunner
from running_modes.scoring.scoring_runner import ScoringRunner
from running_modes.transfer_learning.transfer_learning_runner import TransferLearningRunner
from running_modes.validation.validation_runner import ValidationRunner
from diversity_filters.diversity_filter_factory import DiversityFilterFactory
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from running_modes.enums.running_mode_enum import RunningModeEnum

from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFuncionParameters
from reinvent_scoring.scoring.component_parameters import ComponentParameters



class Manager:

    def __init__(self, configuration):
        self.running_mode_enum = RunningModeEnum()
        self.configuration = GeneralConfigurationEnvelope(**configuration)
        utils_general.set_default_device_cuda()
        self._load_environmental_variables()

    def _run_create_empty_model(self):
        config = CreateModelConfiguration(**self.configuration.parameters)
        runner = CreateModelRunner(self.configuration, config)
        runner.run()

    def _run_transfer_learning(self):
        config = TransferLearningConfiguration(**self.configuration.parameters)
        model = Model.load_from_file(config.input_model_path)
        runner = TransferLearningRunner(model, config, self.configuration)
        runner.run()

    def _run_reinforcement_learning(self):
        rl_components = ReinforcementLearningComponents(**self.configuration.parameters)
        diversity_filter = self._setup_diversity_filter(rl_components.diversity_filter)
        scoring_function = self._setup_scoring_function(rl_components.scoring_function)
        rl_config = ReinforcementLearningConfiguration(**rl_components.reinforcement_learning)
        inception_config = InceptionConfiguration(**rl_components.inception)
        inception = Inception(inception_config, scoring_function, Model.load_from_file(rl_config.prior))
        runner = ReinforcementRunner(self.configuration, rl_config, diversity_filter, scoring_function,
                                     inception=inception)
        runner.run()

    def _setup_diversity_filter(self, diversity_filter_parameters):
        diversity_filter_parameters = DiversityFilterParameters(**diversity_filter_parameters)
        diversity_filter_factory = DiversityFilterFactory()
        diversity_filter = diversity_filter_factory.load_diversity_filter(diversity_filter_parameters)
        return diversity_filter

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

    def _run_curriculum_learning(self):
        runner = CurriculumRunner(self.configuration)
        runner.run()

    def run(self):
        """determines from the configuration object which type of run it is expected to start"""
        switcher = {
            self.running_mode_enum.TRANSFER_LEARNING: self._run_transfer_learning,
            self.running_mode_enum.REINFORCEMENT_LEARNING: self._run_reinforcement_learning,
            self.running_mode_enum.SAMPLING: self._run_sampling,
            self.running_mode_enum.SCORING: self._run_scoring,
            self.running_mode_enum.CREATE_MODEL: self._run_create_empty_model,
            self.running_mode_enum.VALIDATION: self._run_validation,
            self.running_mode_enum.CURRICULUM_LEARNING: self._run_curriculum_learning
        }
        job = switcher.get(self.configuration.run_type, lambda: TypeError)
        job()

    def _load_environmental_variables(self):
        try:
            project_root = os.path.dirname(__file__)
            with open(os.path.join(project_root, '../configs/config.json'), 'r') as f:
                config = json.load(f)
            environmental_variables = config["ENVIRONMENTAL_VARIABLES"]
            for key, value in environmental_variables.items():
                os.environ[key] = value

        except KeyError as ex:
            raise ex
