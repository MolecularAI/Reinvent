class MergingStrategyEnum:
    LINEAR_MERGE = "linear"
    LINEAR_MERGE_WITH_THRESHOLD = "linear_with_threshold"
    LINEAR_MERGE_WITH_BULK = "linear_with_bulk"
    LINEAR_MERGE_WITH_REMOVAL = "linear_with_removal"

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

        # prohibit any attempt to set any values

    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")