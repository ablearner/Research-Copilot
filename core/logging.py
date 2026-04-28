import json
import logging


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter suitable for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            entry.update(extra)
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_format: bool = False) -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        if json_format:
            handler.setFormatter(JsonFormatter())
        else:
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
