import logging
import logging.handlers
import json
from contextvars import ContextVar
from pathlib import Path
from datetime import datetime
from typing import Optional
from config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY

# ── ContextVar ────────────────────────────────────────────────────────────────
# Exported so worker loops and request middleware can set it.
# Each asyncio Task gets its own context copy — setting this inside a task only
# affects that task and all coroutines awaited within it.
batch_id_var: ContextVar[Optional[str]] = ContextVar("batch_id", default=None)


# ── JSON Formatter ────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "function":  record.funcName,
            "line":      record.lineno,
        }
        # Context fields injected by BatchContextFilter
        for field in ("batch_id", "environment", "service"):
            val = getattr(record, field, None)
            if val is not None:
                log_record[field] = val
        # Legacy extra={} dict pattern used by existing callers
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_record.update(record.extra)
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, default=str)


# ── Context Filter ────────────────────────────────────────────────────────────
class BatchContextFilter(logging.Filter):
    """Injects batch_id, environment, and service onto every LogRecord."""

    def __init__(self, environment: str, service: str):
        super().__init__()
        self.environment = environment
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.batch_id    = batch_id_var.get()  # None when no batch context
        record.environment = self.environment
        record.service     = self.service
        return True


# ── CloudWatch Handler Factory ────────────────────────────────────────────────
def _make_cloudwatch_handler(log_group: str, region: str = AWS_REGION):
    """
    Returns a watchtower CloudWatchLogHandler subclass that routes each log
    record to a per-batch_id stream, or "app/general" when no batch context
    is active. Import is deferred so this module loads fine without boto3/watchtower.
    """
    import boto3
    import watchtower

    boto3_client = boto3.client(
        "logs",
        region_name=region,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    class _ContextVarStreamHandler(watchtower.CloudWatchLogHandler):
        def _get_stream_name(self, _message) -> str:
            bid = batch_id_var.get()
            return f"batch/{bid}" if bid else "app/general"

    return _ContextVarStreamHandler(
        log_group_name=log_group,
        log_stream_name="app/general",  # overridden by _get_stream_name above
        boto3_client=boto3_client,
        create_log_group=True,
        create_log_stream=True,
        use_queues=True,
        send_interval=10,
        max_batch_count=1000,
    )


# ── One-time global configuration ────────────────────────────────────────────
_LOGGING_CONFIGURED = False
ROOT_LOGGER_NAME = "dr-agent"


def configure_logging(debug: bool = False) -> None:
    """
    Configure the entire logging stack. Call ONCE from main.py before any
    service module is imported. Safe to call multiple times (idempotent).

    Sets up:
    - "invoice_parser" root logger with console + optional file + optional CW
    - uvicorn/uvicorn.access/uvicorn.error with the same handlers
    - boto3/botocore silenced at WARNING to prevent log→CW→boto→log loops
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    from config import (
        ENVIRONMENT, SERVICE_NAME,
        LOG_CLOUDWATCH_ENABLED, LOG_FILE_ENABLED,
        CW_LOG_GROUP, AWS_REGION,
    )

    level = logging.DEBUG if debug else logging.INFO

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.handlers = []
    root.setLevel(level)
    root.propagate = False

    json_formatter = JsonFormatter()
    context_filter = BatchContextFilter(environment=ENVIRONMENT, service=SERVICE_NAME)
    root.addFilter(context_filter)

    # Console — always on
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(json_formatter)
    root.addHandler(console_handler)

    # File — optional
    if LOG_FILE_ENABLED:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        root.addHandler(file_handler)

    # CloudWatch — optional
    if LOG_CLOUDWATCH_ENABLED:
        try:
            cw_handler = _make_cloudwatch_handler(CW_LOG_GROUP, AWS_REGION)
            cw_handler.setLevel(level)
            cw_handler.setFormatter(json_formatter)
            root.addHandler(cw_handler)
        except Exception as e:
            root.warning(f"CloudWatch handler failed to initialise, continuing without it: {e}")

    # Uvicorn loggers — attach same handlers + filter so access/error logs are
    # also JSON and also land in the correct CloudWatch stream
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers = []
        uv.setLevel(level)
        uv.propagate = False
        for h in root.handlers:
            uv.addHandler(h)
        uv.addFilter(context_filter)

    # Silence noisy AWS internals to prevent recursive CloudWatch log loops
    for noisy in ("boto3", "botocore", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── setup_logger (backward-compatible, now idempotent) ───────────────────────
def setup_logger(name: str = ROOT_LOGGER_NAME, debug: bool = False) -> logging.Logger:
    """
    Returns a named child logger under "invoice_parser". Idempotent — never
    clears or mutates handlers. All 14 existing call sites work unchanged:

      setup_logger()          → getLogger("invoice_parser")
      setup_logger(debug=True)→ same (debug param accepted but ignored here;
                                configure_logging() controls the level)
      setup_logger(__name__)  → getLogger("invoice_parser.<module>")
                                child propagates up, inheriting all handlers
    """
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
