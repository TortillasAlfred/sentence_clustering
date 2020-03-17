import logging
import sys


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = default_formatter()

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def default_formatter():
    return logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
