from dataclasses import dataclass


@dataclass(frozen=True)
class LoggingModeEnum:
    LOCAL = "local"
    REMOTE = "remote"
