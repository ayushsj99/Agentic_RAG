"""
Console-only logger for the document ingestion pipeline.

Logs are printed to the terminal with colour-coded steps and timing info.
Nothing is written to disk — this is purely for developer visibility.
"""

import sys
import logging
import time
from contextlib import contextmanager

# ── Console-only logger ─────────────────────────────────────────────
_logger = logging.getLogger("ingestion_pipeline")
_logger.setLevel(logging.INFO)
_logger.propagate = False          # don't bubble up to root / agent logger

_console = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
)
_console.setFormatter(
    logging.Formatter(
        "\033[36m%(asctime)s\033[0m | \033[33m[INGEST]\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )
)
_logger.addHandler(_console)


# ── Public helpers ───────────────────────────────────────────────────

def log(step: str, message: str) -> None:
    """Log an ingestion step to console only."""
    _logger.info(f"[{step}]  {message}")


def info(message: str) -> None:
    """Quick info line (no step label)."""
    _logger.info(f" {message}")


def success(message: str) -> None:
    """Green-highlighted success message."""
    _logger.info(f"\033[32m >> {message}\033[0m")


def warn(message: str) -> None:
    """Yellow warning."""
    _logger.warning(f"\033[33m !! {message}\033[0m")


def error(message: str) -> None:
    """Red error."""
    _logger.error(f"\033[31m !! {message}\033[0m")


@contextmanager
def timed_step(step: str):
    """Context manager that logs the step start and elapsed time on exit."""
    log(step, "Started")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log(step, f"Completed in {elapsed:.2f}s")
