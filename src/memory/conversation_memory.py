"""
AutoDrive RAG v2.0 — Conversation Memory
Advanced memory management beyond simple sliding window:
  - Summary Memory: condenses old messages into summaries
  - Entity Memory: tracks mentioned cars, preferences, budget
  - User Profile: persists preferences across sessions
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger("chatbot.memory")


class ConversationMemory:
    """
    Multi-layer conversation memory manager.

    Layer 1 — Recent messages (sliding window, last N turns)
    Layer 2 — Summary of older messages (LLM-generated)
    Layer 3 — Entity tracking (cars, preferences, constraints)
    """

    SUMMARIZE_PROMPT = (
        "Summarize the following conversation in 2-3 sentences, "
        "focusing on the user's car preferences and requirements:\n\n"
        "{conversation}"
    )

    def __init__(
        self,
        window_size: int = 10,
        summarize_after: int = 20,
        llm=None,
    ) -> None:
        self.window_size = window_size
        self.summarize_after = summarize_after
        self.llm = llm

        self._messages: dict[str, list[BaseMessage]] = defaultdict(list)
        self._summaries: dict[str, str] = {}
        self._entities: dict[str, dict] = defaultdict(
            lambda: {"cars_mentioned": [], "budget": None, "preferences": {}}
        )

    def add_user_message(self, session_id: str, content: str) -> None:
        self._messages[session_id].append(HumanMessage(content=content))
        self._extract_entities(session_id, content)
        self._maybe_summarize(session_id)

    def add_ai_message(self, session_id: str, content: str) -> None:
        self._messages[session_id].append(AIMessage(content=content))

    def get_messages(
        self, session_id: str, last_n: Optional[int] = None
    ) -> list[BaseMessage]:
        n = last_n or self.window_size
        return self._messages[session_id][-n:]

    def get_context(self, session_id: str) -> dict:
        """Get full memory context: recent messages + summary + entities."""
        return {
            "recent_messages": self.get_messages(session_id),
            "summary": self._summaries.get(session_id, ""),
            "entities": dict(self._entities[session_id]),
        }

    def get_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    def get_entities(self, session_id: str) -> dict:
        return dict(self._entities[session_id])

    def clear(self, session_id: str) -> None:
        self._messages[session_id] = []
        self._summaries.pop(session_id, None)
        self._entities.pop(session_id, None)

    # ── Entity Extraction (rule-based) ──────────────────────────────

    def _extract_entities(self, session_id: str, text: str) -> None:
        entities = self._entities[session_id]
        text_lower = text.lower()

        # Budget extraction
        budget_match = re.search(
            r"(?:budget|under|below|max|within)\s*(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:lakh|lac|l\b|cr)",
            text_lower,
        )
        if budget_match:
            amount = float(budget_match.group(1))
            unit = "cr" if "cr" in text_lower[budget_match.start():budget_match.end()] else "lakh"
            entities["budget"] = amount * (10_000_000 if unit == "cr" else 100_000)

        # Car mentions
        car_id_match = re.findall(r"\bcar\s*(?:id|#|number)?\s*(\d+)", text_lower)
        for cid in car_id_match:
            if cid not in entities["cars_mentioned"]:
                entities["cars_mentioned"].append(cid)

        # Preference extraction
        pref_map = {
            "fuel_type": ["petrol", "diesel", "electric", "cng", "hybrid"],
            "body_type": ["suv", "sedan", "hatchback", "mpv"],
            "transmission": ["automatic", "manual"],
        }
        for pref_key, keywords in pref_map.items():
            for kw in keywords:
                if kw in text_lower:
                    entities["preferences"][pref_key] = kw.capitalize()

        # City preference
        cities = [
            "delhi", "mumbai", "bangalore", "pune", "chennai",
            "hyderabad", "kolkata", "jaipur", "ahmedabad",
        ]
        for city in cities:
            if city in text_lower:
                entities["preferences"]["location"] = city.capitalize()

    def _maybe_summarize(self, session_id: str) -> None:
        msgs = self._messages[session_id]
        if len(msgs) <= self.summarize_after:
            return

        if not self.llm:
            # Simple concatenation fallback
            old = msgs[: -self.window_size]
            text_parts = [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:100]}" for m in old]
            self._summaries[session_id] = "Previous conversation: " + "; ".join(text_parts[-5:])
            return

        try:
            old = msgs[: -self.window_size]
            conv = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}"
                for m in old[-10:]
            )
            prompt = self.SUMMARIZE_PROMPT.format(conversation=conv)
            resp = self.llm.invoke(prompt)
            summary = resp.content if hasattr(resp, "content") else str(resp)
            self._summaries[session_id] = summary.strip()
            logger.info("Summarized %d old messages for session %s", len(old), session_id[:8])
        except Exception as e:
            logger.warning("Summarization failed: %s", e)
