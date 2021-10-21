import json
import os
from abc import ABC, abstractmethod
import logging

import torch
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_summary import FinalSummary

from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLoggerConfiguration


class BaseReinforcementLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope, log_config: ReinforcementLoggerConfiguration):
        self._log_config = log_config
        self._configuration = configuration
        self._setup_workfolder()
        self._logger = self._setup_logger()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, start_time, n_steps, step, score_summary: FinalSummary,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, diversity_filter):
        raise NotImplementedError("timestep_report method is not implemented")

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.result_folder, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def save_checkpoint(self, step: int, diversity_filter, agent):
        actual_step = step + 1
        if self._log_config.logging_frequency > 0 and actual_step % self._log_config.logging_frequency == 0:
            self.save_filter_memory(diversity_filter)
            agent.save_to_file(os.path.join(self._log_config.result_folder, f'Agent.{actual_step}.ckpt'))

    def save_filter_memory(self, diversity_filter):
        diversity_memory = diversity_filter.get_memory_as_dataframe()
        # TODO: Pass a job_name parameter from the config
        self.save_to_csv(diversity_memory, self._log_config.result_folder)

    def save_to_csv(self, scaffold_memory, path, job_name="default_job"):
        sf_enum = ScoringFunctionComponentNameEnum()

        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, "scaffold_memory.csv")

        if len(scaffold_memory) > 0:
            sorted_df = scaffold_memory.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
            sorted_df["ID"] = [f"{job_name}_{e}" for e, _ in enumerate(sorted_df.index.array)]
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
        logger = logging.getLogger("reinforcement_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger