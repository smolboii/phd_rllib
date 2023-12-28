import logging
import sys
import os

def init_logger(log_dir: str) -> logging.Logger:

    # initialises the logging setup

    class LevelFilter(logging.Filter):
        def __init__(self, low, high):
            super().__init__()
            self._low = low
            self._high = high

        def filter(self, record):
            if self._low <= record.levelno <= self._high:
                return True
            return False

    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)

    output_log_path = os.path.join(log_dir, 'output.log')
    error_log_path = os.path.join(log_dir, 'error.log')

    output_fh = logging.FileHandler(output_log_path, mode="a")
    output_fh.addFilter(LevelFilter(0, logging.INFO))
    error_fh = logging.FileHandler(error_log_path, mode="a")
    error_fh.addFilter(LevelFilter(logging.WARNING, logging.CRITICAL))
    sh = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    output_fh.setFormatter(formatter)
    error_fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(output_fh)
    logger.addHandler(error_fh)
    logger.addHandler(sh)

    return logger