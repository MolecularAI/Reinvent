import os
import pandas as pd
import shutil
import unittest

from running_modes.configurations.logging.scoring_log_configuration import ScoringLoggerConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.scoring.scoring_runner_components import ScoringRunnerComponents
from running_modes.configurations.scoring.scoring_runner_configuration import ScoringRunnerConfiguration
from running_modes.scoring.scoring_runner import ScoringRunner
from scoring.component_parameters import ComponentParameters
from scoring.scoring_function_factory import ScoringFunctionFactory
from scoring.scoring_function_parameters import ScoringFuncionParameters
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from utils.enums.logging_mode_enum import LoggingModeEnum
from utils.enums.running_mode_enum import RunningModeEnum
from utils.enums.scoring_runner_enum import ScoringRunnerEnum
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from utils.enums.scoring_function_enum import ScoringFunctionNameEnum


class Test_scoring_runner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lm_enum = LoggingModeEnum()
        run_mode_enum = RunningModeEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        cls.workfolder = MAIN_TEST_PATH
        if not os.path.exists(cls.workfolder):
            os.mkdir(cls.workfolder)
        if not os.path.exists(os.path.join(cls.workfolder, "log")):
            os.mkdir(os.path.join(cls.workfolder, "log"))

        # write some smiles out to a file and also store their tanimoto similarity to the specified smiles below
        smiles = ["CCC", "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N",
                  "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N", "CCXC"]
        path_input = os.path.join(cls.workfolder, "test.smi")

        with open(path_input, "w+") as writer:
            for smile in smiles:
                row = [smile]
                writer.write("{}\n".format("\t".join(row)))
            writer.close()

        # set up tanimoto similarity and custom aller scoring function
        ts_parameters = vars(ComponentParameters(name="tanimoto_similarity", weight=1,
                                                 smiles=["CCC"],
                                                 model_path="", specific_parameters={},
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY))
        matching_substructure = vars(ComponentParameters(component_type=sf_component_enum.MATCHING_SUBSTRUCTURE,
                                                         name="matching_substructure_name",
                                                         weight=1.,
                                                         smiles=["c1ccccc1C"],
                                                         model_path="",
                                                         specific_parameters={}))
        scoring_function = cls._setup_scoring_function(
            ScoringFuncionParameters(name=sf_enum.CUSTOM_SUM, parameters=[ts_parameters, matching_substructure]))

        # set logging
        cls.logging = ScoringLoggerConfiguration(sender="local", recipient=lm_enum.LOCAL,
                                             logging_path=cls.workfolder, job_name="unit_test_job",
                                             job_id="38jsdfilnsdfklj")
        cls.scored_smiles = os.path.join(cls.logging.logging_path, "scored_smiles.csv")

        # do the other fixtures
        scoring_config = ScoringRunnerConfiguration(input=path_input)
        parameters = ScoringRunnerComponents(scoring=vars(scoring_config), scoring_function=vars(scoring_function))
        configuration = GeneralConfigurationEnvelope(parameters=vars(parameters), logging=vars(cls.logging),
                                                     run_type=run_mode_enum.SCORING, version="2.0")
        cls.runner = ScoringRunner(configuration=configuration, config=scoring_config, scoring_function=scoring_function)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.workfolder):
            shutil.rmtree(cls.workfolder)

    @staticmethod
    def _setup_scoring_function(scoring_function_parameters):
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        return scoring_function_instance

    def test_scoring_runner(self):
        RME = ScoringRunnerEnum()
        self.runner.run()
        df = pd.read_csv(filepath_or_buffer=self.scored_smiles, header=0, index_col=None)
        total_score = [round(x, ndigits=3) for x in list(df[RME.TOTAL_SCORE])]
        tanimoto = [round(x, ndigits=3) for x in list(df["tanimoto_similarity"])]
        match = [round(x, ndigits=3) for x in list(df["matching_substructure_name"])]

        self.assertListEqual(total_score, [0.5, 0.034, 0.034, 0.0])
        self.assertListEqual(tanimoto, [1, 0.034, 0.034, 0.0])
        self.assertListEqual(match, [0.5, 1.0, 1.0, 0.0])
        self.assertListEqual(list(df[RME.VALID]), [1, 1, 1, 0])
        self.assertListEqual(["smiles", "total_score", "tanimoto_similarity", "matching_substructure_name", "valid"],
                             list(df.columns))
