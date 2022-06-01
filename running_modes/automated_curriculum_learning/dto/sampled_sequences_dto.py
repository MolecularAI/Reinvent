from dataclasses import dataclass


@dataclass
class SampledSequencesDTO:
    input: str
    output: str
    nll: float
