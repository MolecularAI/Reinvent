from running_modes.configurations.logging.base_log_config import BaseLoggerConfiguration


class ReinforcementLoggerConfiguration(BaseLoggerConfiguration):
    def __init__(self, sender: str, recipient: str, logging_path: str, resultdir: str, logging_frequency=0,
                 job_name="default_name", job_id=None):
        super().__init__(sender=sender, recipient=recipient, logging_path=logging_path, job_name=job_name,
                         job_id=job_id)
        self.logging_frequency = logging_frequency
        self.resultdir = resultdir
