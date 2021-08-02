import time
import json
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_summary import FinalSummary

from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations import ReinforcementLoggerConfiguration


class BaseAutoCLLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        self._configuration = configuration
        self._log_config = ReinforcementLoggerConfiguration(**self._configuration.logging)
        self._setup_workfolder()
        self._logger = self._setup_logger()
        self._start_time = time.time()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, start_time, n_steps, step, smiles,
                        mean_score: np.array, score_summary: FinalSummary, score,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, diversity_filter: BaseDiversityFilter):
        raise NotImplementedError("timestep_report method is not implemented")

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.resultdir, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def log_text_to_file(self, text: str):
        file = os.path.join(self._log_config.logging_path, "curriculum_log.txt")
        with open(file, "a+") as f:
            time_passed = round(time.time() - self._start_time, 4)
            f.write(f"Time from start: {time_passed}s, {text} \n")

    # def log_out_inception(self, inception: Inception):
    #     inception.memory.to_csv(f"{self._log_config.resultdir}/memory.csv")

    def save_checkpoint(self, step, scaffold_filter, agent):
        actual_step = step + 1
        if self._log_config.logging_frequency > 0 and actual_step % self._log_config.logging_frequency == 0:
            self.save_scaffold_memory(scaffold_filter)
            agent.save(os.path.join(self._log_config.resultdir, f'Agent.{actual_step}.ckpt'))

    @abstractmethod
    def save_final_state(self, agent, scaffold_filter):
        raise NotImplementedError("save_final_state method is not implemented")

    def save_merging_state(self, agent, scaffold_filter, name):
        agent.save(os.path.join(self._log_config.resultdir, f'Agent{name}.ckpt'))
        self.save_scaffold_memory(scaffold_filter, memory_name=name)

    def save_scaffold_memory(self, scaffold_filter, memory_name: str = ""):
        scaffold_memory = scaffold_filter.get_memory_as_dataframe()
        self.save_to_csv(scaffold_memory=scaffold_memory, path=self._log_config.resultdir, memory_name=memory_name,
                         job_name=self._log_config.job_name)

    def _setup_workfolder(self):
        if not os.path.isdir(self._log_config.logging_path):
            os.makedirs(self._log_config.logging_path)
        if not os.path.isdir(self._log_config.resultdir):
            os.makedirs(self._log_config.resultdir)

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("auto_cl_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def __del__(self):
        logging.shutdown()

    def save_to_csv(self, scaffold_memory, path: str, memory_name: str = "", job_name: str = "default_job"):
        sf_enum = ScoringFunctionComponentNameEnum()
        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, f"scaffold_memory{memory_name}.csv")

        if len(scaffold_memory) > 0:
            sorted_df = scaffold_memory.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
            sorted_df["ID"] = [f"{job_name}_{e}" for e, _ in enumerate(sorted_df.index.array)]
            sorted_df.to_csv(file_name, index=False)
