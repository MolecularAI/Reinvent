import os
import shutil
import unittest

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters
from reinvent_scoring.scoring.enums.diversity_filter_enum import DiversityFilterEnum
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums.scoring_function_enum import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters

from running_modes.configurations.curriculum_learning.curriculum_learning_components import CurriculumLearningComponents
from running_modes.configurations.curriculum_learning.curriculum_learning_configuration import \
    CurriculumLearningConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.reinforcement_log_configuration import ReinforcementLoggerConfiguration
from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from running_modes.curriculum_learning.curriculum_runner import CurriculumRunner
from running_modes.enums.curriculum_type_enum import CurriculumTypeEnum
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, PRIOR_PATH
from unittest_reinvent.fixtures.test_data import AMOXAPINE, GENTAMICIN, ASPIRIN, CELECOXIB


class TestManualCurriculumLearning(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        self.lm_enum = LoggingModeEnum()
        self.run_mode_enum = RunningModeEnum()
        self.sf_enum = ScoringFunctionNameEnum()
        self.sf_component_enum = ScoringFunctionComponentNameEnum()
        self.filter_enum = DiversityFilterEnum()
        self.scheduled_update_step = 2
        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        smiles = [ASPIRIN, CELECOXIB]
        updated_smiles = [AMOXAPINE, GENTAMICIN]

        self.runner = CurriculumRunner(self._create_configuration(smiles))
        updated_configuration = self._create_configuration(updated_smiles)
        # Write the updated config to disk
        self.runner.logger.log_out_input_configuration(updated_configuration, self.scheduled_update_step)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _create_configuration(self, smiles) -> GeneralConfigurationEnvelope:
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 specific_parameters={"smiles": smiles},
                                                 component_type=self.sf_component_enum.TANIMOTO_SIMILARITY))

        sf_parameters = ScoringFunctionParameters(name=self.sf_enum.CUSTOM_SUM, parameters=[ts_parameters])

        logging = ReinforcementLoggerConfiguration(recipient=self.lm_enum.LOCAL,
                                                   logging_path=self.logging_path, result_folder=self.workfolder,
                                                   logging_frequency=0, job_name="unit_test_job")

        cl_config = CurriculumLearningConfiguration(prior=PRIOR_PATH, agent=PRIOR_PATH,
                                                    update_lock=f'{self.workfolder}/update.lock',
                                                    general_configuration_path=f'{self.workfolder}/input.{self.scheduled_update_step}.json',
                                                    n_steps=3, scheduled_update_step=self.scheduled_update_step)

        scaffold_parameters = DiversityFilterParameters(self.filter_enum.IDENTICAL_MURCKO_SCAFFOLD, 0.05, 25, 0.4)
        inception_config = InceptionConfiguration(smiles, 100, 10)
        parameters = CurriculumLearningComponents(curriculum_learning=vars(cl_config),
                                                  scoring_function=vars(sf_parameters),
                                                  diversity_filter=vars(scaffold_parameters),
                                                  inception=vars(inception_config),
                                                  curriculum_type=CurriculumTypeEnum.MANUAL)
        return GeneralConfigurationEnvelope(parameters=vars(parameters), logging=vars(logging),
                                            run_type=self.run_mode_enum.CURRICULUM_LEARNING, version="3.0")

    def test_manual_curriculum_learning(self):
        self.runner.run()
        self.assertTrue(os.path.isdir(self.logging_path))
