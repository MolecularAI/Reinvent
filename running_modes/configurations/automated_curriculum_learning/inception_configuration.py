from dataclasses import dataclass

from typing import List


@dataclass
class InceptionConfiguration:
    smiles: List[str]
    memory_size: int
    sample_size: int
