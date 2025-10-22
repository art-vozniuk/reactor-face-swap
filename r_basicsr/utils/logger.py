import logging


def get_root_logger(logger_name='basicsr', log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(log_level)
    return logger
