import logging
import logging.handlers
from contextvars import ContextVar
from pathlib import Path
from typing import Optional
from config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY

# ────────────────────────────────────────────────────────────────
# ContextVar (for per-request / per-batch logging)
# ────────────────────────────────────────────────────────────────

batch_id_var: ContextVar[Optional[str]] = ContextVar("batch_id", default=None)

# ────────────────────────────────────────────────────────────────
# Context Filter
# Injects batch_id into log records (if present)
# ────────────────────────────────────────────────────────────────

class BatchContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.batch_id = batch_id_var.get()
        return True

# ────────────────────────────────────────────────────────────────
# CloudWatch Handler Factory (Optional)
# ────────────────────────────────────────────────────────────────

def _make_cloudwatch_handler(log_group: str, region: str = AWS_REGION):
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
        log_stream_name="app/general",
        boto3_client=boto3_client,
        create_log_group=True,
        create_log_stream=True,
        use_queues=True,
        send_interval=10,
        max_batch_count=1000,
    )

# ────────────────────────────────────────────────────────────────
# Global Logging Configuration
# ────────────────────────────────────────────────────────────────

_LOGGING_CONFIGURED = False
ROOT_LOGGER_NAME = "dr-agent"


def configure_logging(debug: bool = False) -> None:
    """
    Call ONCE from main.py before importing service modules.
    Safe to call multiple times (idempotent).
    """

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    from config import (
        LOG_FILE_ENABLED,
        LOG_CLOUDWATCH_ENABLED,
        CW_LOG_GROUP,
        AWS_REGION,
    )

    level = logging.DEBUG if debug else logging.INFO

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.handlers = []
    root.setLevel(level)
    root.propagate = False

    # ─────────────────────────────────────────────
    # Formatter (NON-JSON, readable format)
    # Includes file name + line number
    # ─────────────────────────────────────────────
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - "
        "%(filename)s:%(lineno)d - %(funcName)s() - "
        "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    context_filter = BatchContextFilter()
    root.addFilter(context_filter)

    # ─────────────────────────────────────────────
    # Console Handler
    # ─────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # ─────────────────────────────────────────────
    # File Handler (Optional)
    # ─────────────────────────────────────────────
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
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # ─────────────────────────────────────────────
    # CloudWatch (Optional)
    # ─────────────────────────────────────────────
    if LOG_CLOUDWATCH_ENABLED:
        try:
            cw_handler = _make_cloudwatch_handler(CW_LOG_GROUP, AWS_REGION)
            cw_handler.setLevel(level)
            cw_handler.setFormatter(formatter)
            root.addHandler(cw_handler)
        except Exception as e:
            root.warning(
                f"CloudWatch handler failed to initialise, continuing without it: {e}"
            )

    # ─────────────────────────────────────────────
    # Attach to Uvicorn
    # ─────────────────────────────────────────────
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers = []
        uv.setLevel(level)
        uv.propagate = False
        for h in root.handlers:
            uv.addHandler(h)
        uv.addFilter(context_filter)

    # Silence AWS internals
    for noisy in ("boto3", "botocore", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────────
# Logger Factory
# ────────────────────────────────────────────────────────────────

def setup_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """
    Returns a logger under "dr-agent".

    Usage:
        logger = setup_logger(__name__)
    """

    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)

    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")