from typing import List


class ScoringFuncionParameters:
    def __init__(self, name: str, parameters: List[dict], parallel=False):
        self.name = name
        self.parameters = parameters
        self.parallel = parallel
