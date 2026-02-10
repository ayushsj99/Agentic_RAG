"""
Centralized logger for the Agentic RAG pipeline.

* Every call to `agent_logger.log(step, message)` appends to an in-memory
  list **and** writes to a rotating log file under `backend/logs/`.
* The in-memory list is per-query: call `agent_logger.clear()` before each
  new user query so the UI only sees logs for that turn.
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from threading import Lock

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── File logger (persistent) ────────────────────────────────────────
_file_logger = logging.getLogger("agentic_rag")
_file_logger.setLevel(logging.INFO)

# Rotation: new log file every 2 MB, keep last 5 log files
_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "agent.log"),
    maxBytes=2 * 1024 * 1024,   # 2 MB per file
    backupCount=5,              # agent.log.1 ... agent.log.5
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_file_logger.addHandler(_file_handler)

# ── Console logger (visible in terminal) ───────────────────────────
_console_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
)
_console_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
_file_logger.addHandler(_console_handler)

# ── In-memory log buffer (per-query, shown in the UI) ───────────────
_buffer: list[dict] = []
_lock = Lock()


def log(step: str, message: str) -> None:
    """Log a step from the agent pipeline."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "step": step,
        "message": message,
    }
    with _lock:
        _buffer.append(entry)
    _file_logger.info(f"[{step}] {message}")


def get_logs() -> list[dict]:
    """Return a copy of the current in-memory log buffer."""
    with _lock:
        return list(_buffer)


def clear() -> None:
    """Clear the in-memory buffer (call before each new query)."""
    with _lock:
        _buffer.clear()
