class ROCSInputFileTypesEnum():
    _SHAPE_QUERY = "shape_query"
    _SDF_QUERY = "sdf"
    # Need to stick to this pattern of getter/setter as the alternative more concise one cannot
    # be pickled for multiprocessing

    @property
    def SHAPE_QUERY(self):
        return self._SHAPE_QUERY

    @SHAPE_QUERY.setter
    def SHAPE_QUERY(self, value):
        raise ValueError("Do not assign value to a ROCSInputFileTypesEnum field")

    @property
    def SDF_QUERY(self):
        return self._SDF_QUERY

    @SDF_QUERY.setter
    def SDF_QUERY(self, value):
        raise ValueError("Do not assign value to a ROCSInputFileTypesEnum field")

