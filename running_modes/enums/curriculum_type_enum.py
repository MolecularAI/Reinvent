from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumType:
    AUTOMATED = "automated"
    MANUAL = "manual"

CurriculumTypeEnum = CurriculumType()