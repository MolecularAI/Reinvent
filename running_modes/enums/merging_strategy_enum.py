from dataclasses import dataclass


@dataclass(frozen=True)
class MergingStrategyEnum:
    LINEAR_MERGE = "linear"
    LINEAR_MERGE_WITH_THRESHOLD = "linear_with_threshold"
    LINEAR_MERGE_WITH_BULK = "linear_with_bulk"
    LINEAR_MERGE_WITH_REMOVAL = "linear_with_removal"
