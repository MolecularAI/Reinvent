class BaseLoggerConfiguration:
    def __init__(self, sender: str, recipient: str, logging_path: str, job_name="default_name", job_id=None):
        self.sender = sender
        self.recipient = recipient
        self.logging_path = logging_path
        self.job_name = job_name
        self.job_id = job_id
