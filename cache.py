from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from econobot.config import Settings


class SessionStore(Protocol):
    async def get_session(self, user_id: int) -> dict[str, Any] | None: ...
    async def set_session(self, user_id: int, payload: dict[str, Any]) -> None: ...
    async def delete_session(self, user_id: int) -> None: ...
    async def cleanup_expired(self) -> int: ...


@dataclass
class InMemorySessionStore:
    """
    Non-persistent session store.

    Stores the user's submitted data in a dict in memory (keyed by user_id),
    so no database file is created.
    """

    ttl_s: int

    def __post_init__(self) -> None:
        # user_id -> (last_updated_ts, json_payload)
        self._items: dict[int, tuple[float, str]] = {}

    async def get_session(self, user_id: int) -> dict[str, Any] | None:
        item = self._items.get(user_id)
        if not item:
            return None
        updated_at, raw = item
        if (time.time() - updated_at) > self.ttl_s:
            self._items.pop(user_id, None)
            return None
        return json.loads(raw)

    async def set_session(self, user_id: int, payload: dict[str, Any]) -> None:
        self._items[user_id] = (time.time(), json.dumps(payload, ensure_ascii=False))

    async def delete_session(self, user_id: int) -> None:
        self._items.pop(user_id, None)

    async def cleanup_expired(self) -> int:
        cutoff = time.time() - self.ttl_s
        before = len(self._items)
        self._items = {uid: item for uid, item in self._items.items() if item[0] >= cutoff}
        return before - len(self._items)


async def build_session_store(settings: Settings) -> SessionStore:
    # Always use an in-memory store to avoid creating a database file.
    return InMemorySessionStore(ttl_s=settings.session_ttl_s)

