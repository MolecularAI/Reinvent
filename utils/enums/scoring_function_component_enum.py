

class ScoringFunctionComponentNameEnum():
    __PARALLEL_ROCS_SIMILARITY = "parallel_rocs_similarity"
    __SELECTIVITY = "selectivity"
    __PREDICTIVE_PROPERTY = "predictive_property"
    __ROCS_SIMILARITY = "rocs_similarity"
    __MATCHING_SUBSTRUCTURE = "matching_substructure"
    __TANIMOTO_SIMILARITY = "tanimoto_similarity"
    __TANIMOTO_DISTANCE = "tanimoto_distance"
    __JACCARD_DISTANCE = "jaccard_distance"
    __CUSTOM_ALERTS = "custom_alerts"
    __QED_SCORE = "qed_score"
    __MOLECULAR_WEIGHT = "molecular_weight"
    __NUM_ROTATABLE_BONDS = "num_rotatable_bonds"
    __NUM_HBD_LIPINSKI = "num_hbd_lipinski"
    __NUM_HBA_LIPINSKI = "num_hba_lipinski"
    __NUM_RINGS = "num_rings"
    __TPSA = "tpsa"
    __TOTAL_SCORE = "total_score" # there is no actual component corresponding to this type
    __AZ_LOGD74 = "az_logd74"
    __HLM_CLINT = "hlm_clint"
    __RH_CLINT = "rh_clint"
    __HH_CLINT = "hh_clint"
    __SOLUBILITY_DD = "solubilityDD"
    __CACO2_INTR = "caco2_intr"
    __CACO2_EFFLUX = "caco2_efflux"
    __HERG = "herg"
    __SA_SCORE = "sa_score"
    __AZDOCK = "azdock"

    @property
    def AZDOCK(self):
        return self.__AZDOCK

    @AZDOCK.setter
    def AZDOCK(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def PARALLEL_ROCS_SIMILARITY(self):
        return self.__PARALLEL_ROCS_SIMILARITY

    @PARALLEL_ROCS_SIMILARITY.setter
    def PARALLEL_ROCS_SIMILARITY(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def SELECTIVITY(self):
        return self.__SELECTIVITY

    @SELECTIVITY.setter
    def SELECTIVITY(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def PREDICTIVE_PROPERTY(self):
        return self.__PREDICTIVE_PROPERTY

    @PREDICTIVE_PROPERTY.setter
    def PREDICTIVE_PROPERTY(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def ROCS_SIMILARITY(self):
        return self.__ROCS_SIMILARITY

    @ROCS_SIMILARITY.setter
    def ROCS_SIMILARITY(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def MATCHING_SUBSTRUCTURE(self):
        return self.__MATCHING_SUBSTRUCTURE

    @MATCHING_SUBSTRUCTURE.setter
    def MATCHING_SUBSTRUCTURE(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def TANIMOTO_SIMILARITY(self):
        return self.__TANIMOTO_SIMILARITY

    @TANIMOTO_SIMILARITY.setter
    def TANIMOTO_SIMILARITY(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def TANIMOTO_DISTANCE(self):
        return self.__TANIMOTO_DISTANCE

    @TANIMOTO_DISTANCE.setter
    def TANIMOTO_DISTANCE(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def JACCARD_DISTANCE(self):
        return self.__JACCARD_DISTANCE

    @JACCARD_DISTANCE.setter
    def JACCARD_DISTANCE(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def CUSTOM_ALERTS(self):
        return self.__CUSTOM_ALERTS

    @CUSTOM_ALERTS.setter
    def CUSTOM_ALERTS(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def QED_SCORE(self):
        return self.__QED_SCORE

    @QED_SCORE.setter
    def QED_SCORE(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def MOLECULAR_WEIGHT(self):
        return self.__MOLECULAR_WEIGHT

    @MOLECULAR_WEIGHT.setter
    def MOLECULAR_WEIGHT(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def TPSA(self):
        return self.__TPSA

    @TPSA.setter
    def TPSA(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def NUM_ROTATABLE_BONDS(self):
        return self.__NUM_ROTATABLE_BONDS

    @NUM_ROTATABLE_BONDS.setter
    def NUM_ROTATABLE_BONDS(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def NUM_HBD_LIPINSKI(self):
        return self.__NUM_HBD_LIPINSKI

    @NUM_HBD_LIPINSKI.setter
    def NUM_HBD_LIPINSKI(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def NUM_HBA_LIPINSKI(self):
        return self.__NUM_HBA_LIPINSKI

    @NUM_HBA_LIPINSKI.setter
    def NUM_HBA_LIPINSKI(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def NUM_RINGS(self):
        return self.__NUM_RINGS

    @NUM_RINGS.setter
    def NUM_RINGS(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def TOTAL_SCORE(self):
        return self.__TOTAL_SCORE

    @TOTAL_SCORE.setter
    def TOTAL_SCORE(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def AZ_LOGD74(self):
        return self.__AZ_LOGD74

    @AZ_LOGD74.setter
    def AZ_LOGD74(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def HLM_CLINT(self):
        return self.__HLM_CLINT

    @HLM_CLINT.setter
    def HLM_CLINT(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def RH_CLINT(self):
        return self.__RH_CLINT

    @RH_CLINT.setter
    def RH_CLINT(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def HH_CLINT(self):
        return self.__HH_CLINT

    @HH_CLINT.setter
    def HH_CLINT(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def SOLUBILITY_DD(self):
        return self.__SOLUBILITY_DD

    @SOLUBILITY_DD.setter
    def SOLUBILITY_DD(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def CACO2_INTR(self):
        return self.__CACO2_INTR

    @CACO2_INTR.setter
    def CACO2_INTR(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def CACO2_EFFLUX(self):
        return self.__CACO2_EFFLUX

    @CACO2_EFFLUX.setter
    def CACO2_EFFLUX(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def HERG(self):
        return self.__HERG

    @HERG.setter
    def HERG(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")

    @property
    def SA_SCORE(self):
        return self.__SA_SCORE

    @SA_SCORE.setter
    def SA_SCORE(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionComponentNameEnum field")