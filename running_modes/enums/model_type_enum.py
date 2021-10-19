from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTypeEnum:
    DEFAULT = "default"
    LIB_INVENT = "lib_invent"
    LINK_INVENT = "link_invent"
