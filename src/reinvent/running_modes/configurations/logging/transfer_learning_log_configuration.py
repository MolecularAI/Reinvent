from .base_log_config import BaseLoggerConfiguration


class TransferLearningLoggerConfig(BaseLoggerConfiguration):
    def __init__(self, sender: str, recipient: str, logging_path: str, job_name="default_name", use_weights=False,
                 job_id=None):
        super().__init__(sender=sender, recipient=recipient, logging_path=logging_path, job_name=job_name,
                         job_id=job_id)
        self.use_weights = use_weights
