"""
Logging utility for the Annual Report RAG project.

Every module should obtain its logger via get_logger(__name__)
instead of using print() or configuring logging directly.
"""

import logging

from src.config.settings import LOG_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


if __name__ == "__main__":
    test_logger = get_logger(__name__)
    test_logger.info("Logger smoke test started for %s", __name__)
    test_logger.info("Integer placeholder test value=%d", 1)
    test_logger.info("Float placeholder test value=%f", 1.0)
    test_logger.info("Logger smoke test completed for %s", __name__)
