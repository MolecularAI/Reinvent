from dataclasses import dataclass
from typing import List
from pydantic import Field


@dataclass
class InceptionConfiguration:
    memory_size: int
    sample_size: int
    smiles: List[str] = Field(default_factory=list)
    # inputs: List[str] = Field(default_factory=list)

