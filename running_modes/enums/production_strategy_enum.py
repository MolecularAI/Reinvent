from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionStrategyEnum:
    STANDARD = "standard"
    SPECIFIC_COMPONENTS = "specific_components"