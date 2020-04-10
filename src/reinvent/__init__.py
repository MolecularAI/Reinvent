from .data import REINVENT_DATA_DIRECTORY
from .running_modes.manager import Manager


def run(configuration):
    manager = Manager(configuration)
    manager.run()
