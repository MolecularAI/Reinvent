from dataclasses import dataclass

from typing import List, Dict


@dataclass
class MergingStrategyParametersConfiguration:
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120

    special_parameters: dict = None

@dataclass
class ProductionStrategyParametersConfiguration:
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120
    n_steps: int = 100

    special_parameters: dict = None

@dataclass
class RankingStrategyParametersConfiguration:
    learning_rate: float = 0.0001
    batch_size: int = 64
    sigma: float = 120

    special_parameters: dict = None

