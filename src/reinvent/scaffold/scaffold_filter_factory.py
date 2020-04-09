from .identical_murcko_scaffold import IdenticalMurckoScaffold
from .identical_topological_scaffold import IdenticalTopologicalScaffold
from .scaffold_similarity import ScaffoldSimilarity
from .no_scaffold_filter import NoScaffoldFilter
from .scaffold_parameters import ScaffoldParameters


class ScaffoldFilterFactory:

    def __init__(self):
        self.__scaffold_registry = self.__default_scaffold_registry()

    def __default_scaffold_registry(self) -> dict:
        scaffold_list = dict(IdenticalMurckoScaffold=IdenticalMurckoScaffold,
                             IdenticalTopologicalScaffold=IdenticalTopologicalScaffold,
                             ScaffoldSimilarity=ScaffoldSimilarity,
                             NoFilter=NoScaffoldFilter)
        return scaffold_list

    def load_scaffold_filter(self, scaffold_parameters: ScaffoldParameters):
        scaffold_filter = self.__scaffold_registry[scaffold_parameters.name]
        return scaffold_filter(scaffold_parameters)
