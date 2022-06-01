from dataclasses import dataclass


@dataclass(frozen=True)
class LearningStrategyEnum:
    DAP = "dap"
    MAULI = "mauli"
    MASCOF = "mascof"
    SDAP = "sdap"
    DAP_SINGLE_QUERY = "dap_single_query"
