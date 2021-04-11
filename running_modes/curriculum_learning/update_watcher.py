import json
import os
import time

from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.configurations.curriculum_learning import CurriculumLearningComponents, \
    CurriculumLearningConfiguration


class UpdateWatcher:

    def __init__(self, runner):
        self.runner = runner

    def check_for_update(self, step):
        if os.path.isfile(self.runner.config.update_lock):
            with open(self.runner.config.general_configuration_path) as file:
                sigma = self.runner.config.sigma
                json_input = file.read().replace('\r', '').replace('\n', '')
                configuration = json.loads(json_input)
                self.runner.envelope = GeneralConfigurationEnvelope(**configuration)
                config_components = CurriculumLearningComponents(**self.runner.envelope.parameters)
                self.runner.config = CurriculumLearningConfiguration(**config_components.curriculum_learning)
                # NOTE: We are keeping sigma unchanged
                self.runner.config.sigma = sigma

                self.runner.scoring_function = self.runner.setup_scoring_function(config_components.scoring_function)
                self.runner.logger.save_diversity_memory_checkpoint(self.runner.diversity_filter, step)
                self.runner.diversity_filter = self.runner._setup_diversity_filter(config_components.diversity_filter) #FIXME <== bad practice
                self.runner.inception = self.runner.setup_inception(config_components.inception)
                # self._margin_guard.update_widnow_start(step)
                self.runner.logger.log_message(f"updating the run parameters at step {step}")
                self.runner.logger.log_out_input_configuration(self.runner.envelope, step)
            os.remove(self.runner.config.update_lock)

    def check_for_pause(self):
        """Can be used for pausing the runner for a user defined interval of seconds"""
        if os.path.isfile(self.runner.config.pause_lock):
            pause_limit = self.runner.config.pause_limit

            while pause_limit > 0:
                self.runner.logger.log_message(f"Pausing for {pause_limit} seconds !")
                time.sleep(1.0)
                pause_limit -= 1

                if not os.path.isfile(self.runner.config.pause_lock):
                    pause_limit = 0

            if os.path.isfile(self.runner.config.pause_lock):
                os.remove(self.runner.config.pause_lock)

    def check_for_scheduled_update(self, step: int):
        if self.runner.config.scheduled_update_step == step:
            with open(self.runner.config.update_lock, 'a'):
                os.utime(self.runner.config.update_lock, None)
