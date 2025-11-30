import logging
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger once for the app.
    """
    # Avoid double-configuring in case this is called multiple times
    if logging.getLogger().handlers:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a module-level logger.
    """
    return logging.getLogger(name or __name__)
