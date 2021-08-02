class ScoringTableEnum:

    AGENTS = "agents"
    SCORES = "scores"
    SCORING_FUNCTIONS = "scoring_functions"
    COMPONENT_NAMES = "component_names"

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

        # prohibit any attempt to set any values

    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")