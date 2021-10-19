import json
import os

from running_modes.create_model.logging.base_create_model_logger import BaseCreateModelLogger


class LocalCreateModelLogger(BaseCreateModelLogger):
    def log_message(self, message: str):
        self._common_logger.info(message)


