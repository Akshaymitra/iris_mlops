import logging
import sys

def setup_logger(log_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Log to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optionally log to a file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
