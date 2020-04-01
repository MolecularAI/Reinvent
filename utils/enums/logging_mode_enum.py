class LoggingModeEnum():
    _LOCAL = "local"
    _REMOTE = "remote"

    @property
    def LOCAL(self):
        return self._LOCAL

    @LOCAL.setter
    def LOCAL(self, value):
        raise ValueError("Do not assign value to a LoggingModeEnum field")

    @property
    def REMOTE(self):
        return self._REMOTE

    @REMOTE.setter
    def REMOTE(self, value):
        raise ValueError("Do not assign value to a LoggingModeEnum field")

