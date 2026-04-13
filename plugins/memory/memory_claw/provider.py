"""MemoryClawProvider — MemoryProvider implementation for Memory-Claw.

Implements the MemoryProvider ABC for the hermes-agent plugin system.
Exposes 5 tools (mclaw_store, mclaw_recall, mclaw_forget, mclaw_stats, mclaw_update)
and provides background prefetch, auto-extraction, session-end summary,
and pre-compress insight preservation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent.memory_provider import MemoryProvider

from .store import MemoryStore
from .embedder import MistralEmbedder
from .extractor import MemoryExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

STORE_SCHEMA = {
    "name": "mclaw_store",
    "description": (
        "Store a memory for future recall. Use when the user states a preference, "
        "makes a decision, shares a fact, or provides technical details worth "
        "remembering across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The fact or information to remember.",
            },
            "category": {
                "type": "string",
                "description": "Category: preference, decision, entity, technical, seo, workflow, debug, fact.",
                "default": "fact",
            },
            "importance": {
                "type": "number",
                "description": "Importance score 0-1 (default: 0.5). Values >= 0.85 are core memories.",
                "default": 0.5,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization.",
            },
        },
        "required": ["text"],
    },
}

RECALL_SCHEMA = {
    "name": "mclaw_recall",
    "description": (
        "Search memory for relevant past facts, preferences, and decisions. "
        "Returns results ranked by weighted score (similarity + importance + recency). "
        "Use when you need context about the user's past behavior or preferences."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default: 5, max: 20).",
                "default": 5,
            },
            "category": {
                "type": "string",
                "description": "Optional category filter.",
            },
            "tier": {
                "type": "string",
                "description": "Optional tier filter.",
            },
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "mclaw_forget",
    "description": (
        "Delete a specific memory by its ID. Use sparingly — this is permanent. "
        "Get memory IDs from mclaw_recall results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The ID of the memory to delete.",
            },
        },
        "required": ["memory_id"],
    },
}

STATS_SCHEMA = {
    "name": "mclaw_stats",
    "description": (
        "Get statistics about the memory store: total memories, count by tier "
        "(core/contextual/episodic), and count by category."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

UPDATE_SCHEMA = {
    "name": "mclaw_update",
    "description": "Update an existing memory's text, importance, category, or tags.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of the memory to update."},
            "text": {"type": "string", "description": "New text content (optional, leave empty to keep current)."},
            "importance": {"type": "number", "description": "New importance score 0-1 (optional)."},
            "category": {"type": "string", "description": "New category (optional)."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags (optional, replaces existing)."},
        },
        "required": ["memory_id"],
    },
}

ALL_TOOL_SCHEMAS = [STORE_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA, STATS_SCHEMA, UPDATE_SCHEMA]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MemoryClawProvider(MemoryProvider):
    """LanceDB-backed semantic memory with Mistral embeddings."""

    def __init__(self):
        self._store: Optional[MemoryStore] = None
        self._embedder: Optional[MistralEmbedder] = None
        self._extractor: Optional[MemoryExtractor] = None
        self._db_path: str = ""
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._last_prefetch_query: str = ""
        self._last_prefetch_vector: Optional[List[float]] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "memory_claw"

    def is_available(self) -> bool:
        """Check if MISTRAL_API_KEY is configured. No network calls."""
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if api_key:
            return True
        # Check $HERMES_HOME/.env
        hermes_home = os.environ.get("HERMES_HOME", "")
        if hermes_home:
            env_path = Path(hermes_home) / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("MISTRAL_API_KEY="):
                        return bool(line.split("=", 1)[1].strip().strip("\"'"))
        return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "Mistral API key for embeddings",
                "secret": True,
                "env_var": "MISTRAL_API_KEY",
                "url": "https://console.mistral.ai/api-keys",
                "required": True,
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize store and embedder."""
        hermes_home = kwargs.get("hermes_home", os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
        self._db_path = str(Path(hermes_home) / "memory-claw")

        # Ensure MISTRAL_API_KEY is available
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            env_path = Path(hermes_home) / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("MISTRAL_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("\"'")
                        os.environ["MISTRAL_API_KEY"] = api_key
                        break

        if not api_key:
            logger.debug("memory-claw: MISTRAL_API_KEY not found — plugin inactive")
            return

        try:
            self._embedder = MistralEmbedder(api_key=api_key)
            self._extractor = MemoryExtractor(api_key=api_key)
            self._store = MemoryStore(self._db_path)
            self._store.open()
            self._initialized = True
            logger.info("memory-claw: initialized at %s", self._db_path)
        except Exception as e:
            logger.warning("memory-claw: failed to initialize: %s", e)

    def system_prompt_block(self) -> str:
        """Return system prompt text."""
        if not self._initialized:
            return ""
        return (
            "# Memory Claw\n"
            "Active. Use mclaw_store to save facts, mclaw_recall to search memory, "
            "mclaw_forget to delete memories, mclaw_stats for statistics, "
            "mclaw_update to edit existing memories."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched context from background thread."""
        if not self._initialized:
            return ""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Memory Claw Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background search for the upcoming turn.

        Always includes all core-tier memories plus semantic search results.
        Skips if the query is very similar to the previous one (cosine sim > 0.9).
        Minimum score threshold: 0.3.
        """
        if not self._initialized or not query:
            return

        def _run():
            try:
                vector = self._embedder.embed(query)
                if not vector:
                    return

                # Skip if query is nearly identical to the previous one
                if self._last_prefetch_vector:
                    last = np.array(self._last_prefetch_vector)
                    curr = np.array(vector)
                    norm_product = np.linalg.norm(last) * np.linalg.norm(curr)
                    if norm_product > 0:
                        cosine_sim = float(np.dot(last, curr) / norm_product)
                        if cosine_sim > 0.9:
                            logger.debug("memory-claw: prefetch skipped (similar query)")
                            return

                self._last_prefetch_vector = vector

                # Semantic search results
                results = self._store.search(vector, limit=5)
                # Filter by minimum score
                results = [r for r in results if r.get("score", 0) >= 0.3]

                # Always include all core-tier memories
                core_memories = self._store.get_all_core()
                core_ids = {m["id"] for m in core_memories}
                result_ids = {r["id"] for r in results}

                # Merge: results first (ordered by score), then core not already in results
                merged = list(results)
                for cm in core_memories:
                    if cm["id"] not in result_ids:
                        merged.append(cm)

                if not merged:
                    return

                lines = []
                for r in merged:
                    cat = r.get("category", "")
                    src = r.get("source", "")
                    tier = r.get("tier", "")
                    score = r.get("score", "")
                    score_str = f" (score:{score})" if score else ""
                    lines.append(
                        f"- [{tier}][{cat}] {r['text']}{score_str}"
                    )
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug("memory-claw prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mclaw-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Auto-extract facts from each conversation turn."""
        if not self._initialized or not user_content:
            return

        def _extract_and_store():
            try:
                facts = self._extractor.extract_from_turn(user_content, assistant_content)
                for fact in facts:
                    text = fact.get("text", "")
                    if not text or not text.strip():
                        continue
                    vector = self._embedder.embed(text.strip())
                    if not vector:
                        continue
                    # Check duplicates
                    dup_id = self._store.find_duplicates(vector, threshold=0.82)
                    if dup_id:
                        logger.debug("sync_turn: duplicate fact skipped: %s", text[:50])
                        continue
                    self._store.add(
                        text=text.strip(),
                        vector=vector,
                        importance=float(fact.get("importance", 0.5)),
                        category=fact.get("category", "fact"),
                        tags=fact.get("tags", []),
                        source="auto",
                    )
            except Exception as e:
                logger.debug("memory-claw sync_turn failed: %s", e)

        t = threading.Thread(target=_extract_and_store, daemon=True, name="mclaw-sync")
        t.start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to memory-claw."""
        if action != "add" or not content:
            return
        if not self._initialized:
            return

        def _write():
            try:
                vector = self._embedder.embed(content)
                if vector:
                    self._store.add(
                        text=content,
                        vector=vector,
                        importance=0.6,
                        category="fact",
                        source=f"builtin-{target}",
                    )
            except Exception as e:
                logger.debug("memory-claw memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="mclaw-memwrite")
        t.start()

    def on_session_end(self, messages: list) -> None:
        """Extract key conclusions at session end."""
        if not self._initialized or not messages:
            return

        def _extract():
            try:
                facts = self._extractor.extract_session_summary(messages)
                for fact in facts:
                    text = fact.get("text", "")
                    if not text or not text.strip():
                        continue
                    vector = self._embedder.embed(text.strip())
                    if not vector:
                        continue
                    dup_id = self._store.find_duplicates(vector, threshold=0.82)
                    if dup_id:
                        continue
                    self._store.add(
                        text=text.strip(),
                        vector=vector,
                        importance=float(fact.get("importance", 0.6)),
                        category=fact.get("category", "session_summary"),
                        tags=fact.get("tags", ["session-end"]),
                        source="session_summary",
                    )
            except Exception as e:
                logger.debug("memory-claw on_session_end failed: %s", e)

        t = threading.Thread(target=_extract, daemon=True, name="mclaw-session-end")
        t.start()

    def on_pre_compress(self, messages: list) -> str:
        """Extract insights before context compression."""
        if not self._initialized or not messages:
            return ""
        try:
            return self._extractor.extract_pre_compress_insights(messages)
        except Exception as e:
            logger.debug("memory-claw on_pre_compress failed: %s", e)
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas."""
        if not self._initialized:
            return []
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a memory-claw tool call."""
        if not self._initialized:
            return json.dumps({"error": "Memory Claw is not active for this session."})

        try:
            if tool_name == "mclaw_store":
                text = args.get("text", "")
                if not text or not text.strip():
                    return json.dumps({"error": "Missing required parameter: text"})

                importance = max(0.0, min(1.0, float(args.get("importance", 0.5))))
                category = args.get("category", "fact")
                tags = args.get("tags", [])
                confidence = float(args.get("confidence", 0.6))
                user_id = args.get("user_id", "default")
                relations = args.get("relations", [])
                metadata_str = args.get("metadata", "{}")

                vector = self._embedder.embed(text.strip())
                if not vector:
                    return json.dumps({"error": "Failed to generate embedding."})

                # Check for duplicates
                dup_id = self._store.find_duplicates(vector, threshold=0.7)
                if dup_id:
                    return json.dumps({
                        "result": f"Similar memory already exists (id: {dup_id}). Not stored as duplicate.",
                        "duplicate_id": dup_id,
                    })

                mem_id = self._store.add(
                    text=text.strip(),
                    vector=vector,
                    importance=importance,
                    category=category,
                    tags=tags,
                    source="tool",
                    confidence=confidence,
                    user_id=user_id,
                    relations=relations,
                    metadata=metadata_str,
                )
                if mem_id:
                    return json.dumps({
                        "result": f"Memory stored (id: {mem_id}, importance: {importance:.2f})",
                        "id": mem_id,
                    })
                return json.dumps({"error": "Failed to store memory."})

            elif tool_name == "mclaw_recall":
                query = args.get("query", "")
                if not query or not query.strip():
                    return json.dumps({"error": "Missing required parameter: query"})
                limit = max(1, min(int(args.get("limit", 5)), 20))

                vector = self._embedder.embed(query.strip())
                if not vector:
                    return json.dumps({"error": "Failed to generate embedding."})

                results = self._store.search(
                    vector,
                    limit=limit,
                    tier_filter=args.get("tier"),
                    category_filter=args.get("category"),
                )

                if not results:
                    return json.dumps({"result": "No relevant memories found."})

                formatted = []
                for r in results:
                    formatted.append({
                        "id": r["id"],
                        "text": r["text"],
                        "tier": r["tier"],
                        "category": r["category"],
                        "importance": round(r["importance"], 2),
                        "score": r["score"],
                        "source": r.get("source", ""),
                        "hit_count": r.get("hit_count", 0),
                        "confidence": r.get("confidence", 0.6),
                        "user_id": r.get("user_id", "default"),
                        "relations": r.get("relations", []),
                        "metadata": r.get("metadata", "{}"),
                    })

                return json.dumps({"result": formatted, "count": len(formatted)})

            elif tool_name == "mclaw_forget":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    return json.dumps({"error": "Missing required parameter: memory_id"})
                mem = self._store.get_by_id(memory_id)
                if not mem:
                    return json.dumps({"error": f"Memory {memory_id} not found."})
                success = self._store.delete(memory_id)
                if success:
                    return json.dumps({"result": f"Memory {memory_id} deleted."})
                return json.dumps({"error": f"Failed to delete memory {memory_id}."})

            elif tool_name == "mclaw_stats":
                stats = self._store.get_stats()
                return json.dumps({"result": stats})

            elif tool_name == "mclaw_update":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    return json.dumps({"error": "Missing required parameter: memory_id"})

                new_text = args.get("text")
                new_importance = args.get("importance")
                new_category = args.get("category")
                new_tags = args.get("tags")

                # If text is changing, re-embed
                if new_text and new_text.strip():
                    vector = self._embedder.embed(new_text.strip())
                    if not vector:
                        return json.dumps({"error": "Failed to generate embedding for updated text."})

                success = self._store.update(
                    memory_id=memory_id,
                    text=new_text,
                    importance=float(new_importance) if new_importance is not None else None,
                    category=new_category,
                    tags=new_tags,
                )
                if success:
                    return json.dumps({"result": f"Memory {memory_id} updated."})
                return json.dumps({"error": f"Failed to update memory {memory_id}. Not found or update error."})

            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.error("memory-claw tool %s failed: %s", tool_name, e)
            return json.dumps({"error": f"memory-claw {tool_name} failed: {e}"})

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
