import logging


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root_logger.addHandler(handler)

    for logger_name in (
        "apps",
        "core",
        "agents",
        "retrieval",
        "services",
        "adapters",
        "rag_runtime",
    ):
        logging.getLogger(logger_name).setLevel(resolved_level)
