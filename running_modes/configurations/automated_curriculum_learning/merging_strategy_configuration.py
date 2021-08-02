from dataclasses import dataclass

from running_modes.configurations.automated_curriculum_learning.strategy_parameters_configuration import \
    MergingStrategyParametersConfiguration


@dataclass
class MergingStrategyConfiguration:
    # merging strategy name
    name: str
    # merging sf, df, inception shared with ranking
    scoring_function: dict
    diversity_filter: dict
    inception: dict
    max_num_iterations: int
    parameters: MergingStrategyParametersConfiguration


