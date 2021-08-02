from typing import Optional
from pydantic import BaseModel


class BaseLoggerConfiguration(BaseModel):
    recipient: str
    logging_path: str
    sender: str = ""
    job_name: str = "default_name"
    job_id: Optional[str] = None
