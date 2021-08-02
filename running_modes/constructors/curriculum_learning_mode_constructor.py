from running_modes.utils.general import set_default_device_cuda
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.curriculum_learning.curriculum_runner import CurriculumRunner



class CurriculumLearningModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        set_default_device_cuda()
        runner = CurriculumRunner(self._configuration)
        return runner