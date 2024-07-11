import logging
import os
from pathlib import Path


class Logger:
    def __init__(self, folder_path):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.folder_path = folder_path
        self._initialize_logger()

    def _initialize_logger(self):
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        # File Handler
        fileHandler = logging.FileHandler(Path(self.folder_path) / "logger.log")
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)

        # Console Handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)
