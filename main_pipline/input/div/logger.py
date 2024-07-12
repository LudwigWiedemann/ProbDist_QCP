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
        self.fileHandler = logging.FileHandler(Path(self.folder_path) / "logger.log")
        self.fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(self.fileHandler)

        # Console Handler
        self.consoleHandler = logging.StreamHandler()
        self.consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(self.consoleHandler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def close(self):
        self.fileHandler.close()
        self.consoleHandler.close()
        self.logger.removeHandler(self.fileHandler)
        self.logger.removeHandler(self.consoleHandler)
