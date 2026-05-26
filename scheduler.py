from __future__ import annotations

import logging
from typing import Any

import aiocron

from econobot.config import Settings

log = logging.getLogger(__name__)


async def start_cleanup_jobs(bot: Any, settings: Settings, session_store: Any) -> None:
    # Runs every 5 minutes. Keeps it simple & safe for both Redis and SQLite.
    @aiocron.crontab("*/5 * * * *")
    async def _cleanup() -> None:
        try:
            removed = await session_store.cleanup_expired()
            if removed:
                log.info("Cleaned up %s expired sessions", removed)
        except Exception:
            log.exception("Cleanup job failed")

