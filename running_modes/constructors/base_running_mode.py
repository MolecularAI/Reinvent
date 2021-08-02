from abc import abstractmethod, ABC


class BaseRunningMode(ABC):

    @abstractmethod
    def run(self):
        raise NotImplementedError("run method is not implemented")