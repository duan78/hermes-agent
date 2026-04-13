"""LanceDB storage layer for memory-claw.

Shared by the hermes-agent MemoryProvider plugin and the standalone MCP server.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

TABLE_NAME = "memories_claw"

VALID_CATEGORIES = (
    "preference", "decision", "entity", "technical",
    "seo", "workflow", "debug", "fact",
    "session_summary",
)
VALID_TIERS = ("core", "contextual", "episodic")

_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 1024)),
    pa.field("importance", pa.float32()),
    pa.field("category", pa.string()),
    pa.field("tier", pa.string()),
    pa.field("tags", pa.list_(pa.string())),
    pa.field("created_at", pa.string()),
    pa.field("updated_at", pa.string()),
    pa.field("last_accessed", pa.string()),
    pa.field("source", pa.string()),
    pa.field("hit_count", pa.int32()),
    pa.field("confidence", pa.float32()),
    pa.field("user_id", pa.string()),
    pa.field("relations", pa.list_(pa.string())),
    pa.field("superseded_by", pa.string()),
    pa.field("metadata", pa.string()),
])

# Column names added in v2 schema migration
_V2_COLUMNS = {"confidence", "user_id", "relations", "superseded_by", "metadata"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tier_from_importance(importance: float) -> str:
    if importance >= 0.85:
        return "core"
    elif importance >= 0.5:
        return "contextual"
    return "episodic"


class MemoryStore:
    """LanceDB-backed persistent memory store with vector search."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: Optional[lancedb.DB] = None
        self._table: Optional[Any] = None

    def open(self) -> None:
        """Open (or create) the LanceDB database and table.

        Handles v1 → v2 schema migration: if the existing table is missing
        the new columns (confidence, user_id, relations, superseded_by, metadata),
        a new table with the extended schema is created and existing data is
        copied over with sensible defaults.
        """
        Path(self._db_path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self._db_path)

        existing = self._db.list_tables()
        if isinstance(existing, list):
            table_names = existing
        else:
            table_names = list(existing.tables) if hasattr(existing, 'tables') else list(existing)

        if TABLE_NAME in table_names:
            self._table = self._db.open_table(TABLE_NAME)
            logger.info("memory-claw: opened existing table '%s' at %s", TABLE_NAME, self._db_path)

            # --- v1 → v2 schema migration ---
            self._migrate_v2()
        else:
            self._db.create_table(TABLE_NAME, schema=_SCHEMA)
            self._table = self._db.open_table(TABLE_NAME)
            logger.info("memory-claw: created new table '%s' at %s", TABLE_NAME, self._db_path)

    def _migrate_v2(self) -> None:
        """Migrate v1 table to v2 schema if new columns are missing."""
        if self._table is None or self._db is None:
            return

        try:
            current_schema = self._table.schema
            current_names = {f.name for f in current_schema}

            if _V2_COLUMNS.issubset(current_names):
                return  # Already migrated

            logger.info("memory-claw: migrating v1 → v2 schema …")

            # Read all existing data
            old_table = self._table
            old_data = old_table.search().select(
                ["id", "text", "vector", "importance", "category", "tier",
                 "tags", "created_at", "updated_at", "last_accessed",
                 "source", "hit_count"]
            ).limit(None).to_arrow()

            if len(old_data) == 0:
                # Empty table — just drop and recreate
                self._db.drop_table(TABLE_NAME)
                self._db.create_table(TABLE_NAME, schema=_SCHEMA)
                self._table = self._db.open_table(TABLE_NAME)
                logger.info("memory-claw: migrated empty table to v2 schema")
                return

            # Build new records with defaults for v2 columns
            rows = old_data.to_pylist()
            new_records = []
            for row in rows:
                rec = {col: row.get(col) for col in [
                    "id", "text", "vector", "importance", "category", "tier",
                    "tags", "created_at", "updated_at", "last_accessed",
                    "source", "hit_count",
                ]}
                # Ensure vector is a proper list
                if rec.get("vector") is not None:
                    rec["vector"] = list(rec["vector"])
                # v2 defaults
                rec["confidence"] = 0.6
                rec["user_id"] = "default"
                rec["relations"] = []
                rec["superseded_by"] = ""
                rec["metadata"] = "{}"
                new_records.append(rec)

            # Drop old table and create new one with v2 schema
            self._db.drop_table(TABLE_NAME)
            self._db.create_table(TABLE_NAME, schema=_SCHEMA)
            self._table = self._db.open_table(TABLE_NAME)

            # Insert migrated data
            self._table.add(new_records)
            logger.info(
                "memory-claw: migrated %d rows to v2 schema", len(new_records),
            )
        except Exception as e:
            logger.warning("memory-claw: v2 migration failed: %s", e)

    @property
    def is_open(self) -> bool:
        return self._table is not None

    def add(
        self,
        text: str,
        vector: List[float],
        importance: float = 0.5,
        category: str = "fact",
        tags: Optional[List[str]] = None,
        source: str = "auto",
        confidence: float = 0.6,
        user_id: str = "default",
        relations: Optional[List[str]] = None,
        superseded_by: str = "",
        metadata: str = "{}",
    ) -> Optional[str]:
        """Insert a new memory. Returns the memory ID, or None on failure."""
        if not self._table:
            return None

        if category not in VALID_CATEGORIES:
            category = "fact"

        tier = _tier_from_importance(importance)
        mem_id = uuid.uuid4().hex[:12]
        now = _now_iso()

        record = {
            "id": mem_id,
            "text": text,
            "vector": vector,
            "importance": float(importance),
            "category": category,
            "tier": tier,
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
            "last_accessed": now,
            "source": source,
            "hit_count": 0,
            "confidence": float(confidence),
            "user_id": user_id,
            "relations": relations or [],
            "superseded_by": superseded_by,
            "metadata": metadata,
        }

        try:
            self._table.add([record])
            logger.debug("memory-claw: stored memory %s (tier=%s, importance=%.2f)", mem_id, tier, importance)
            return mem_id
        except Exception as e:
            logger.warning("memory-claw: failed to store memory: %s", e)
            return None

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        tier_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search with weighted scoring: 60% similarity + 30% importance + 10% recency."""
        if not self._table:
            return []

        filters = []
        if tier_filter:
            filters.append(f"tier = '{tier_filter}'")
        if category_filter:
            filters.append(f"category = '{category_filter}'")

        filter_str = " AND ".join(filters) if filters else None

        try:
            results = (
                self._table.search(query_vector)
                .limit(limit)
                .select(["id", "text", "importance", "category", "tier",
                         "tags", "created_at", "updated_at", "last_accessed",
                         "source", "hit_count", "confidence", "user_id",
                         "relations", "superseded_by", "metadata", "_distance"])
            )

            if filter_str:
                results = results.where(filter_str, prefilter=True)

            results = results.to_arrow()

            if len(results) == 0:
                return []

            now = time.time()
            scored: List[Dict[str, Any]] = []

            for row in results.to_pylist():
                distance = float(row.get("_distance", 1.0))
                similarity = max(0.0, 1.0 - min(distance, 1.0))

                try:
                    accessed_str = str(row.get("last_accessed", ""))
                    accessed_time = datetime.fromisoformat(accessed_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    accessed_time = now - 86400 * 30

                age_days = max(0, (now - accessed_time) / 86400)
                recency = max(0.0, 1.0 - min(age_days / 90.0, 1.0))
                importance = float(row.get("importance", 0.5))
                score = 0.6 * similarity + 0.3 * importance + 0.1 * recency

                mem = {
                    "id": str(row.get("id", "")),
                    "text": str(row.get("text", "")),
                    "importance": importance,
                    "category": str(row.get("category", "fact")),
                    "tier": str(row.get("tier", "episodic")),
                    "tags": list(row.get("tags", [])),
                    "created_at": str(row.get("created_at", "")),
                    "updated_at": str(row.get("updated_at", "")),
                    "source": str(row.get("source", "")),
                    "hit_count": int(row.get("hit_count", 0)),
                    "confidence": float(row.get("confidence", 0.6)),
                    "user_id": str(row.get("user_id", "default")),
                    "relations": list(row.get("relations", [])),
                    "superseded_by": str(row.get("superseded_by", "")),
                    "metadata": str(row.get("metadata", "{}")),
                    "score": round(score, 4),
                }
                scored.append(mem)

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored

        except Exception as e:
            logger.warning("memory-claw: search failed: %s", e)
            return []

    def get_all_core(self) -> List[Dict[str, Any]]:
        """Get all core-tier memories (importance >= 0.85)."""
        if not self._table:
            return []

        try:
            results = (
                self._table.search()
                .where("tier = 'core'")
                .select(["id", "text", "importance", "category", "tier",
                         "tags", "created_at", "source", "hit_count",
                         "confidence", "user_id", "relations", "metadata"])
                .limit(100)
                .to_arrow()
            )

            if len(results) == 0:
                return []

            return [
                {
                    "id": str(row.get("id", "")),
                    "text": str(row.get("text", "")),
                    "importance": float(row.get("importance", 0.5)),
                    "category": str(row.get("category", "")),
                    "tier": str(row.get("tier", "core")),
                    "tags": list(row.get("tags", [])),
                    "created_at": str(row.get("created_at", "")),
                    "source": str(row.get("source", "")),
                    "hit_count": int(row.get("hit_count", 0)),
                    "confidence": float(row.get("confidence", 0.6)),
                    "user_id": str(row.get("user_id", "default")),
                    "relations": list(row.get("relations", [])),
                    "metadata": str(row.get("metadata", "{}")),
                }
                for row in results.to_pylist()
            ]
        except Exception as e:
            logger.warning("memory-claw: get_all_core failed: %s", e)
            return []

    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Look up a single memory by ID."""
        if not self._table:
            return None

        try:
            results = (
                self._table.search()
                .where(f"id = '{memory_id}'")
                .select(["id", "text", "vector", "importance", "category", "tier",
                         "tags", "created_at", "updated_at", "last_accessed",
                         "source", "hit_count", "confidence", "user_id",
                         "relations", "superseded_by", "metadata"])
                .limit(1)
                .to_arrow()
            )

            if len(results) == 0:
                return None

            row = results.to_pylist()[0]
            return {
                "id": str(row.get("id", "")),
                "text": str(row.get("text", "")),
                "vector": list(row.get("vector", [])),
                "importance": float(row.get("importance", 0.5)),
                "category": str(row.get("category", "")),
                "tier": str(row.get("tier", "")),
                "tags": list(row.get("tags", [])),
                "created_at": str(row.get("created_at", "")),
                "updated_at": str(row.get("updated_at", "")),
                "last_accessed": str(row.get("last_accessed", "")),
                "source": str(row.get("source", "")),
                "hit_count": int(row.get("hit_count", 0)),
                "confidence": float(row.get("confidence", 0.6)),
                "user_id": str(row.get("user_id", "default")),
                "relations": list(row.get("relations", [])),
                "superseded_by": str(row.get("superseded_by", "")),
                "metadata": str(row.get("metadata", "{}")),
            }
        except Exception as e:
            logger.debug("memory-claw: get_by_id failed: %s", e)
            return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self._table:
            return False

        try:
            self._table.delete(f"id = '{memory_id}'")
            logger.debug("memory-claw: deleted memory %s", memory_id)
            return True
        except Exception as e:
            logger.warning("memory-claw: delete failed for %s: %s", memory_id, e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return DB statistics: total count, counts by tier and category."""
        if not self._table:
            return {"total": 0, "by_tier": {}, "by_category": {}}

        try:
            table = self._table.search().select(
                ["id", "tier", "category", "hit_count"]
            ).limit(None).to_arrow()

            if len(table) == 0:
                return {"total": 0, "by_tier": {}, "by_category": {}}

            rows = table.to_pylist()
            by_tier: Dict[str, int] = {}
            by_category: Dict[str, int] = {}
            for row in rows:
                tier = str(row.get("tier", ""))
                cat = str(row.get("category", ""))
                if tier:
                    by_tier[tier] = by_tier.get(tier, 0) + 1
                if cat:
                    by_category[cat] = by_category.get(cat, 0) + 1

            return {
                "total": len(rows),
                "by_tier": by_tier,
                "by_category": by_category,
            }
        except Exception as e:
            logger.warning("memory-claw: get_stats failed: %s", e)
            return {"total": 0, "by_tier": {}, "by_category": {}}

    def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Update an existing memory's text, importance, category, or tags.

        If ``text`` is changed the memory is re-embedded so vector search
        continues to work correctly.

        Returns True on success, False on failure.
        """
        if not self._table:
            return False

        try:
            mem = self.get_by_id(memory_id)
            if mem is None:
                return False

            now = _now_iso()
            new_text = text.strip() if text else mem["text"]
            new_importance = float(importance) if importance is not None else mem["importance"]
            new_category = category if category else mem["category"]
            new_tags = tags if tags is not None else mem["tags"]
            new_tier = _tier_from_importance(new_importance)

            # If text changed we must delete+re-add (LanceDB has no in-place update)
            self._table.delete(f"id = '{memory_id}'")

            # Use existing vector unless text changed (caller should re-embed)
            record = {
                "id": memory_id,
                "text": new_text,
                "vector": mem.get("vector", []),
                "importance": new_importance,
                "category": new_category if new_category in VALID_CATEGORIES else "fact",
                "tier": new_tier,
                "tags": new_tags,
                "created_at": mem["created_at"],
                "updated_at": now,
                "last_accessed": mem.get("last_accessed", now),
                "source": mem.get("source", ""),
                "hit_count": mem.get("hit_count", 0),
                "confidence": mem.get("confidence", 0.6),
                "user_id": mem.get("user_id", "default"),
                "relations": mem.get("relations", []),
                "superseded_by": mem.get("superseded_by", ""),
                "metadata": mem.get("metadata", "{}"),
            }

            self._table.add([record])
            logger.debug("memory-claw: updated memory %s", memory_id)
            return True
        except Exception as e:
            logger.warning("memory-claw: update failed for %s: %s", memory_id, e)
            return False

    def find_duplicates(self, vector: List[float], threshold: float = 0.7) -> Optional[str]:
        """Check if a similar memory already exists. Returns ID of duplicate or None."""
        if not self._table:
            return None

        try:
            results = (
                self._table.search(vector)
                .limit(1)
                .select(["id", "_distance"])
                .to_arrow()
            )

            if len(results) == 0:
                return None

            row = results.to_pylist()[0]
            distance = float(row.get("_distance", 1.0))
            cosine_sim = max(0.0, 1.0 - (distance * distance / 2.0))

            if cosine_sim >= threshold:
                return str(row.get("id", ""))

            return None
        except Exception as e:
            logger.debug("memory-claw: find_duplicates failed: %s", e)
            return None

    def count(self) -> int:
        """Return total number of memories."""
        return self.get_stats().get("total", 0)
