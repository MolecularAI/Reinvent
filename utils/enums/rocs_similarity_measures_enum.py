class ROCSSimilarityMeasuresEnum():
    _TANIMOTO = "Tanimoto"
    _REF_TVERSKY = "RefTversky"
    _FIT_TVERSKY = "FitTversky"

    # Need to stick to this pattern of getter/setter as the alternative more concise one cannot
    # be pickled for multiprocessing

    @property
    def TANIMOTO(self):
        return self._TANIMOTO

    @TANIMOTO.setter
    def TANIMOTO(self, value):
        raise ValueError("Do not assign value to a ROCSSimiliarityMeasuresEnum field")

    @property
    def REF_TVERSKY(self):
        return self._REF_TVERSKY

    @REF_TVERSKY.setter
    def REF_TVERSKY(self, value):
        raise ValueError("Do not assign value to a ROCSSimiliarityMeasuresEnum field")

    @property
    def FIT_TVERSKY(self):
        return self._FIT_TVERSKY

    @FIT_TVERSKY.setter
    def FIT_TVERSKY(self, value):
        raise ValueError("Do not assign value to a ROCSSimiliarityMeasuresEnum field")
