

class ScaffoldFilterEnum():
    __IDENTICAL_TOPOLOGICAL_SCAFFOLD = "IdenticalTopologicalScaffold"
    __IDENTICAL_MURCKO_SCAFFOLD = "IdenticalMurckoScaffold"
    __SCAFFOLD_SIMILARITY = "ScaffoldSimilarity"
    __NO_FILTER = "NoFilter"

    @property
    def IDENTICAL_TOPOLOGICAL_SCAFFOLD(self):
        return self.__IDENTICAL_TOPOLOGICAL_SCAFFOLD

    @IDENTICAL_TOPOLOGICAL_SCAFFOLD.setter
    def IDENTICAL_TOPOLOGICAL_SCAFFOLD(self, value):
        raise ValueError("Do not assign value to a ScaffoldFilterEnum field")

    @property
    def IDENTICAL_MURCKO_SCAFFOLD(self):
        return self.__IDENTICAL_MURCKO_SCAFFOLD

    @IDENTICAL_MURCKO_SCAFFOLD.setter
    def IDENTICAL_MURCKO_SCAFFOLD(self, value):
        raise ValueError("Do not assign value to a ScaffoldFilterEnum field")

    @property
    def SCAFFOLD_SIMILARITY(self):
        return self.__SCAFFOLD_SIMILARITY

    @SCAFFOLD_SIMILARITY.setter
    def SCAFFOLD_SIMILARITY(self, value):
        raise ValueError("Do not assign value to a ScaffoldFilterEnum field")

    @property
    def NO_FILTER(self):
        return self.__NO_FILTER

    @NO_FILTER.setter
    def NO_FILTER(self, value):
        raise ValueError("Do not assign value to a ScaffoldFilterEnum field")