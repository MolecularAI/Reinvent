from typing import List
from dataclasses import dataclass


@dataclass
class ComponentParameters:
    component_type: str
    name: str
    weight: float
    smiles: List[str]
    model_path: str
    specific_parameters: dict = None
