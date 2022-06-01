from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStrategyEnum:
    USER_DEFINED = "user_defined"
    LINK_INVENT = "link_invent"
    NO_CURRICULUM = "no_curriculum"
