"""Central logging setup for CloudLens.

Every module gets a logger via `get_logger(__name__)`. This module configures
a shared "cloudlens" logger hierarchy with two handlers:

- console (INFO+): a clean, concise narrative of the flow — user query, which
  tool was called with what arguments, what came back. Meant to be readable
  while watching the terminal live.
- per-session file (DEBUG+) under logs/: the same narrative plus full JSON
  payloads, for after-the-fact debugging.

Log lines are tagged with the active session id via a contextvar, so adapters
and tools never need a session_id parameter threaded through their
signatures — they just log normally and the active session's file handler
(filtered by session id) picks it up. `asyncio.to_thread` and
`run_coroutine_threadsafe` both propagate contextvars, so this works
transparently across the Chainlit UI's background-thread tool execution.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

LOGS_DIR = Path(__file__).resolve().parent / "logs"

_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")

_CONSOLE_FORMAT = "%(asctime)s [%(session_id)s] %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(session_id)s] %(name)s: %(message)s"
_TIME_FORMAT = "%H:%M:%S"

_configured = False


class _SessionFilter(logging.Filter):
    """Stamps every record with the active session id from the contextvar."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = _session_id_var.get()
        return True


def _configure_root() -> logging.Logger:
    global _configured
    root = logging.getLogger("cloudlens")
    if _configured:
        return root

    root.setLevel(logging.DEBUG)

    # Chainlit's CLI calls logging.basicConfig() at import time, which attaches
    # its own handler to Python's true root logger. Without this, our records
    # would propagate up and also print via Chainlit's handler — in Chainlit's
    # format, without our session tag, and duplicated alongside our own line.
    root.propagate = False

    # NOTE: the session filter must live on each HANDLER, not on the "cloudlens"
    # logger itself. Logger-level filters (Logger.addFilter) only run on the
    # record's *originating* logger (e.g. "cloudlens.agent.loop"), not on
    # ancestor loggers during propagation — so a filter added here would never
    # fire. Handler-level filters run for every record that reaches that
    # handler, regardless of which child logger it came from.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_TIME_FORMAT))
    console.addFilter(_SessionFilter())
    root.addHandler(console)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Module-level logger, e.g. get_logger(__name__). Safe to call at import time."""
    _configure_root()
    return logging.getLogger(f"cloudlens.{name}")


def set_current_session(session_id: str) -> None:
    """Activate `session_id` for logging in the calling task/thread.

    Call this at the top of any handler that resumes a session started
    elsewhere (e.g. Chainlit's per-message on_message, given the session id
    stored earlier in cl.user_session) — Chainlit dispatches each event as
    its own task, so a contextvar set in on_chat_start does not automatically
    carry into a later on_message call.
    """
    _session_id_var.set(session_id)


def start_session(label: str = "session") -> tuple[str, logging.FileHandler]:
    """Begin a new logged session: assigns a session id, creates its log file
    under logs/, attaches a filtered file handler, and activates the session
    for the calling task. Returns (session_id, handler) — pass both to
    end_session() when the session is over, and pass session_id to
    set_current_session() to resume logging under it from another task.
    """
    root = _configure_root()
    LOGS_DIR.mkdir(exist_ok=True)

    session_id = uuid.uuid4().hex[:8]
    set_current_session(session_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{timestamp}_{label}_{session_id}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=None))
    handler.addFilter(_SessionFilter())
    handler.addFilter(lambda record: getattr(record, "session_id", None) == session_id)

    root.addHandler(handler)
    root.info("=== SESSION START [%s] log=%s ===", label, log_path.name)
    return session_id, handler


def end_session(session_id: str, handler: logging.FileHandler) -> None:
    root = logging.getLogger("cloudlens")
    set_current_session(session_id)
    root.info("=== SESSION END ===")
    root.removeHandler(handler)
    handler.close()


@contextmanager
def session_scope(label: str = "session") -> Iterator[str]:
    """Convenience context manager for single-shot entry points (the CLI)."""
    session_id, handler = start_session(label)
    try:
        yield session_id
    finally:
        end_session(session_id, handler)
