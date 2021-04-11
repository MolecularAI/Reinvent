from diversity_filters import IdenticalMurckoScaffold, IdenticalTopologicalScaffold, ScaffoldSimilarity, NoScaffoldFilter
from diversity_filters.base_diversity_filter import BaseDiversityFilter
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters


class DiversityFilterFactory:

    def __init__(self):
        self.__scaffold_registry = self._default_scaffold_registry()

    def _default_scaffold_registry(self) -> dict:
        scaffold_list = dict(IdenticalMurckoScaffold=IdenticalMurckoScaffold,
                             IdenticalTopologicalScaffold=IdenticalTopologicalScaffold,
                             ScaffoldSimilarity=ScaffoldSimilarity,
                             NoFilter=NoScaffoldFilter)
        return scaffold_list

    def load_diversity_filter(self, scaffold_parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        diversity_filter = self.__scaffold_registry[scaffold_parameters.name]
        return diversity_filter(scaffold_parameters)
