import os


def get_remote_logging_auth_token():
    return os.getenv('REMOTE_LOGGING_AUTH_TOKEN')
