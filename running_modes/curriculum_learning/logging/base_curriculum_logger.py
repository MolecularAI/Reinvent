import json
import logging
import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from reinvent_scoring import ComponentSpecificParametersEnum

from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.reinforcement_log_configuration import ReinforcementLoggerConfiguration
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_summary import FinalSummary


class BaseCurriculumLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self._configuration = configuration
        self._log_config = ReinforcementLoggerConfiguration(**self._configuration.logging)
        self._setup_workfolder()
        self._logger = self._setup_logger()
        self._specific_parameters_enum = ComponentSpecificParametersEnum()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, start_time, n_steps, step, smiles,
                        mean_score: np.array, score_summary: FinalSummary, score,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, diversity_filter):
        raise NotImplementedError("timestep_report method is not implemented")

    def log_out_input_configuration(self, configuration: GeneralConfigurationEnvelope, step=0):
        output_file = os.path.join(self._log_config.result_folder, f"input.{step}.json")
        jsonstr = json.dumps(configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(output_file, 'w') as f:
            f.write(jsonstr)

    def log_out_inception(self, inception: Inception):
        inception.memory.to_csv(f"{self._log_config.result_folder}/memory.csv")

    def save_checkpoint(self, step, scaffold_filter, agent):
        actual_step = step + 1
        if self._log_config.logging_frequency > 0 and actual_step % self._log_config.logging_frequency == 0:
            self.save_diversity_memory(scaffold_filter)
            agent.save_to_file(os.path.join(self._log_config.result_folder, f'Agent.{actual_step}.ckpt'))

    @abstractmethod
    def save_final_state(self, agent, scaffold_filter):
        raise NotImplementedError("save_final_state method is not implemented")

    def save_diversity_memory(self, diversity_filter):
        diversity_memory = diversity_filter.get_memory_as_dataframe()
        self.save_to_csv(diversity_memory, self._log_config.result_folder, self._log_config.job_name)

    def save_diversity_memory_checkpoint(self, diversity_filter, step):
        if not os.path.isdir(self._log_config.logging_path):
            os.makedirs(self._log_config.logging_path)
        sf_enum = ScoringFunctionComponentNameEnum()

        diversity_memory = diversity_filter.get_memory_as_dataframe()
        file_name = os.path.join(self._log_config.logging_path, f"{step}_scaffold_memory.csv")

        if len(diversity_memory) > 0:
            sorted_df = diversity_memory.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
            sorted_df["ID"] = [f"{self._log_config.job_name}_{e}" for e, _ in enumerate(sorted_df.index.array)]
            sorted_df.to_csv(file_name, index=False)

    def _setup_workfolder(self):
        if not os.path.isdir(self._log_config.logging_path):
            os.makedirs(self._log_config.logging_path)
        if not os.path.isdir(self._log_config.result_folder):
            os.makedirs(self._log_config.result_folder)

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("curriculum_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def __del__(self):
        logging.shutdown()

    def save_to_csv(self, scaffold_memory, path, job_name="default_job"):

        sf_enum = ScoringFunctionComponentNameEnum()
        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, "scaffold_memory.csv")

        if len(scaffold_memory) > 0:
            sorted_df = scaffold_memory.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
            sorted_df["ID"] = [f"{job_name}_{e}" for e, _ in enumerate(sorted_df.index.array)]
            sorted_df.to_csv(file_name, index=False)
