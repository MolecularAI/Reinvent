import json
import logging
import os
from abc import ABC, abstractmethod

from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum

from running_modes.automated_curriculum_learning.dto.timestep_dto import TimestepDTO
from running_modes.configurations import GeneralConfigurationEnvelope, CurriculumLoggerConfiguration


class BaseLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope, log_config: CurriculumLoggerConfiguration):
        self._configuration = configuration
        self._log_config = log_config
        self._setup_workfolder()
        self._logger = self._setup_logger()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, report_dto: TimestepDTO, diversity_filter, agent):
        raise NotImplementedError("timestep_report method is not implemented")

    @abstractmethod
    def save_final_state(self, agent, diversity_filter):
        raise NotImplementedError("save_final_state method is not implemented")

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

    def save_merging_state(self, agent, diversity_filter, name):
        agent.save_to_file(os.path.join(self._log_config.result_folder, f'Agent{name}.ckpt'))
        self.save_filter_memory(diversity_filter, memory_name=name)

    def save_filter_memory(self, diversity_filter, memory_name: str = ""):
        diversity_memory = diversity_filter.get_memory_as_dataframe()
        self._save_to_csv(diversity_memory, self._log_config.result_folder, memory_name, self._log_config.job_name)

    def _save_to_csv(self, diversity_memory, path: str, memory_name: str = "", job_name: str = "default_job"):
        sf_enum = ScoringFunctionComponentNameEnum()
        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, f"scaffold_memory{memory_name}.csv")

        if len(diversity_memory) > 0:
            sorted_df = diversity_memory.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
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
        logger = logging.getLogger("curriculum_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
