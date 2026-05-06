"""
AutoDrive Chatbot — Chat History Manager
Provides in-memory history for local dev, Redis for production.
"""

from __future__ import annotations
from collections import defaultdict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from .config import settings


class InMemoryHistory:
    """Simple dict-backed message store for local development."""

    def __init__(self) -> None:
        self._store: dict[str, list[BaseMessage]] = defaultdict(list)

    def get_messages(self, session_id: str, last_n: int = 10) -> list[BaseMessage]:
        return self._store[session_id][-last_n:]

    def add_user_message(self, session_id: str, content: str) -> None:
        self._store[session_id].append(HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str) -> None:
        self._store[session_id].append(AIMessage(content=content))

    def clear(self, session_id: str) -> None:
        self._store[session_id] = []


class RedisHistory:
    """Redis-backed message store for production (Azure / Docker)."""

    def __init__(self) -> None:
        from langchain_community.chat_message_histories import (
            RedisChatMessageHistory,
        )
        self._history_cls = RedisChatMessageHistory

    def _get_history(self, session_id: str):
        return self._history_cls(session_id, url=settings.REDIS_URL)

    def get_messages(self, session_id: str, last_n: int = 10) -> list[BaseMessage]:
        h = self._get_history(session_id)
        return h.messages[-last_n:]

    def add_user_message(self, session_id: str, content: str) -> None:
        self._get_history(session_id).add_user_message(content)

    def add_ai_message(self, session_id: str, content: str) -> None:
        self._get_history(session_id).add_ai_message(content)

    def clear(self, session_id: str) -> None:
        self._get_history(session_id).clear()


def get_history_store() -> InMemoryHistory | RedisHistory:
    """Factory — returns Redis store if configured, else in-memory."""
    if settings.has_redis:
        return RedisHistory()
    return InMemoryHistory()
