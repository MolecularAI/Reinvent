from pydantic import BaseModel


class BaseConfiguration(BaseModel):
    curriculum_type: str