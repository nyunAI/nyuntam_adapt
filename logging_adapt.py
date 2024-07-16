import logging
import os
import shutil
from datetime import datetime
import sys


class LogFile(object):
    """File-like object to log text using the `logging` module."""

    def __init__(self, name=None):
        self.logger = logging.getLogger(name)

    def write(self, msg, level=logging.INFO):
        self.logger.log(level, msg)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()

def define_logger(name, logging_path):
    logger = logging.getLogger(name)

    logging.basicConfig(
        filename=f"{logging_path}/log.log",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H-%M-%S",
        level=logging.INFO,
    )
    sys.stdout = LogFile("stdout")

    return logger
