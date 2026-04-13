"""LLM-based fact and entity extraction for memory-claw.

Uses Mistral-small-latest to extract structured facts, entities, and
insights from conversation turns, sessions, and pre-compress context.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_EXTRACT_TURN_SYSTEM = """\
You are a memory extraction assistant. Analyze the conversation turn below
and extract facts worth remembering for future sessions.

RULES:
- Skip greetings, confirmations, acknowledgments, and trivial chatter.
- Only extract genuinely useful information: preferences, decisions, facts,
  technical details, identities, workflow steps, debugging insights.
- Each fact must be a self-contained sentence that makes sense without context.
- Assign importance: 0.85+ for identity/core facts (name, role, key preferences),
  0.5-0.84 for important operational facts, <0.5 for minor details.
- Category must be one of: preference, decision, entity, technical, seo,
  workflow, debug, fact, session_summary.
- Generate 2-3 relevant tags per fact.

Return a JSON array of extracted facts. Each element must have:
  {"text": str, "category": str, "importance": float, "tags": [str]}

If nothing worth remembering, return an empty array: []"""

_EXTRACT_SESSION_SYSTEM = """\
You are a memory extraction assistant. Analyze the full conversation session
below and extract key conclusions and takeaways worth remembering.

RULES:
- Focus on conclusions reached, decisions made, problems solved, and
  important facts learned during this session.
- Each fact must be self-contained and useful in future sessions.
- Assign importance: 0.85+ for core conclusions, 0.5-0.84 for important
  takeaways, <0.5 for minor observations.
- Category should typically be "session_summary" or "decision".
- Generate 2-3 relevant tags per fact.

Return a JSON array: [{"text": str, "category": str, "importance": float, "tags": [str]}]
If nothing worth remembering, return []."""

_EXTRACT_COMPRESS_SYSTEM = """\
You are a memory preservation assistant. The messages below are about to be
compressed (summarized) to save context space. Extract any important facts,
decisions, or context that should be preserved.

Return a concise bulleted list of key points that should be injected into
the compression summary. If nothing important would be lost, return an empty string."""

_EXTRACT_ENTITIES_SYSTEM = """\
You are an entity extraction assistant. Extract named entities from the text.
Focus on: people, organizations, projects, tools, frameworks, products, domains.

Return a JSON array of entity name strings. Example: ["OpenAI", "React", "ProjectX"]
If no entities found, return []."""


class MemoryExtractor:
    """Extracts facts, entities, and insights from conversation turns."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.mistral.ai/v1",
        model: str = "mistral-small-latest",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def _chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call the Mistral chat API. Returns the assistant message content or None."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.debug("MemoryExtractor: LLM timeout")
            return None
        except Exception as e:
            logger.debug("MemoryExtractor: LLM call failed: %s", e)
            return None

    @staticmethod
    def _parse_json_array(raw: Optional[str]) -> list:
        """Best-effort parse of a JSON array from LLM output."""
        if not raw:
            return []
        # Strip markdown code fences if present
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            # Try to find a JSON array in the text
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass
        return []

    def extract_from_turn(self, user_content: str, assistant_content: str) -> list:
        """Extract facts from a single conversation turn.

        Returns list of {"text": str, "category": str, "importance": float, "tags": list[str]}.
        Returns empty list if nothing worth remembering.
        """
        if not user_content and not assistant_content:
            return []

        prompt = (
            f"<user>\n{user_content}\n</user>\n"
            f"<assistant>\n{assistant_content}\n</assistant>"
        )

        raw = self._chat(_EXTRACT_TURN_SYSTEM, prompt)
        facts = self._parse_json_array(raw)

        # Validate and normalize each fact
        validated = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            text = f.get("text", "")
            if not text or not isinstance(text, str) or not text.strip():
                continue
            validated.append({
                "text": text.strip(),
                "category": str(f.get("category", "fact")),
                "importance": max(0.0, min(1.0, float(f.get("importance", 0.5)))),
                "tags": list(f.get("tags", [])) if isinstance(f.get("tags"), list) else [],
            })
        return validated

    def extract_session_summary(self, messages: list) -> list:
        """Extract key conclusions from a full session.

        Args:
            messages: list of {"role": str, "content": str} dicts.

        Returns list of {"text": str, "category": str, "importance": float, "tags": list[str]}.
        """
        if not messages:
            return []

        # Build a condensed transcript (limit to avoid token overflow)
        parts = []
        for msg in messages[-40:]:  # last 40 messages max
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:500]
            if content:
                parts.append(f"[{role}]: {content}")

        if not parts:
            return []

        prompt = "Conversation session:\n" + "\n".join(parts)
        raw = self._chat(_EXTRACT_SESSION_SYSTEM, prompt)
        facts = self._parse_json_array(raw)

        validated = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            text = f.get("text", "")
            if not text or not isinstance(text, str) or not text.strip():
                continue
            validated.append({
                "text": text.strip(),
                "category": str(f.get("category", "session_summary")),
                "importance": max(0.0, min(1.0, float(f.get("importance", 0.6)))),
                "tags": list(f.get("tags", ["session-end"])) if isinstance(f.get("tags"), list) else ["session-end"],
            })
        return validated

    def extract_pre_compress_insights(self, messages: list) -> str:
        """Extract important insights from messages about to be compressed.

        Returns formatted text for injection into compression summary.
        """
        if not messages:
            return ""

        parts = []
        for msg in messages[-30:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:400]
            if content:
                parts.append(f"[{role}]: {content}")

        if not parts:
            return ""

        prompt = "Messages to be compressed:\n" + "\n".join(parts)
        raw = self._chat(_EXTRACT_COMPRESS_SYSTEM, prompt)

        if raw and raw.strip():
            return f"## Preserved Memory Context\n{raw.strip()}"
        return ""

    def extract_entities(self, text: str) -> list:
        """Extract named entities from text.

        Returns list of entity name strings.
        """
        if not text or not text.strip():
            return []

        raw = self._chat(_EXTRACT_ENTITIES_SYSTEM, text)
        entities = self._parse_json_array(raw)

        # Validate: only keep string entries
        return [str(e).strip() for e in entities if isinstance(e, str) and e.strip()]
