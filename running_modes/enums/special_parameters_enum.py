class SpecialParametersEnum:
    THRESHOLD_RANKING = "threshold_ranking"
    THRESHOLD_MERGING = "threshold_merging"
    MAX_NUM_ITERATIONS = "max_num_iteration"
    BULK_SIZE = "bulk_size"
    TIME_LIMIT = "time_limit"
    SELECTED_COMPONENTS = "selected_components"
    # specific parameters used for ordering components in user_defined_order strategy for ranking.
    SPECIFIC_PARAMETERS = "specific_parameters"
    ORDER = "order"
    # for individual helper component merging thresholds
    MERGING_THRESHOLD = "merging_threshold"

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

        # prohibit any attempt to set any values

    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")