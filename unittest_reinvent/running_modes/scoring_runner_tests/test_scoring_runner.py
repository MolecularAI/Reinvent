import os
import shutil
import unittest

import pandas as pd
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums.scoring_function_enum import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.scoring_log_configuration import ScoringLoggerConfiguration
from running_modes.configurations.scoring.scoring_runner_components import ScoringRunnerComponents
from running_modes.configurations.scoring.scoring_runner_configuration import ScoringRunnerConfiguration
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.enums.scoring_runner_enum import ScoringRunnerEnum
from running_modes.scoring.scoring_runner import ScoringRunner
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from unittest_reinvent.fixtures.test_data import PROPANE, ASPIRIN, COCAINE, BENZENE, CAFFEINE


class TestScoringRunner(unittest.TestCase):

    def setUp(self):
        lm_enum = LoggingModeEnum()
        run_mode_enum = RunningModeEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        self.workfolder = MAIN_TEST_PATH
        if not os.path.exists(self.workfolder):
            os.mkdir(self.workfolder)

        # write some smiles out to a file and also store their tanimoto similarity to the specified smiles below
        smiles = [PROPANE, COCAINE, ASPIRIN, BENZENE]
        path_input = os.path.join(self.workfolder, "test.smi")

        with open(path_input, "w+") as f:
            for smile in smiles:
                row = [smile]
                f.write("{}\n".format("".join(row)))

        # set up tanimoto similarity and custom aller scoring function
        ts_parameters = vars(ComponentParameters(name="tanimoto_similarity", weight=1,
                                                 specific_parameters={"smiles":[PROPANE, ASPIRIN]},
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY))
        matching_substructure = vars(ComponentParameters(component_type=sf_component_enum.MATCHING_SUBSTRUCTURE,
                                                         name="matching_substructure_name",
                                                         weight=1.,
                                                         specific_parameters={"smiles":[CAFFEINE]}))
        scoring_function_parameters = ScoringFunctionParameters(name=sf_enum.CUSTOM_SUM, parameters=[ts_parameters, matching_substructure])
        scoring_function = self._setup_scoring_function(scoring_function_parameters)

        # set utils
        self.logging = ScoringLoggerConfiguration(recipient=lm_enum.LOCAL,
                                                  logging_path=self.workfolder, job_name="unit_test_job",
                                                  job_id="")
        self.scored_smiles = os.path.join(self.workfolder, "scored_smiles.csv")

        # do the other fixtures
        scoring_config = ScoringRunnerConfiguration(input=path_input)
        parameters = ScoringRunnerComponents(scoring=scoring_config, scoring_function=scoring_function_parameters)
        configuration = GeneralConfigurationEnvelope(parameters=vars(parameters), logging=vars(self.logging),
                                                     run_type=run_mode_enum.SCORING, version="2.0")
        self.runner = ScoringRunner(configuration=configuration, config=scoring_config, scoring_function=scoring_function)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

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

        self.assertListEqual(total_score, [0.5, 0.095, 0.5, 0.122])
        self.assertListEqual(tanimoto, [1.0, 0.19, 1.0, 0.245])
        self.assertListEqual(match, [0.5, 0.5, 0.5, 0.5])
        self.assertListEqual(list(df[RME.VALID]), [1, 1, 1, 1])
        self.assertListEqual(["smiles", "total_score", "tanimoto_similarity", "matching_substructure_name", "valid"],
                             list(df.columns))
