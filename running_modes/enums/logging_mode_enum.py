class LoggingModeEnum:
    LOCAL = "local"
    REMOTE = "remote"

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    def __setattr__(self, key, value):
        raise ValueError("Do not assign value to a LoggingModeEnum field.")

