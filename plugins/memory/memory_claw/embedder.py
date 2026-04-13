"""Mistral embedding client with LRU cache for memory-claw.

Shared by the hermes-agent MemoryProvider plugin and the standalone MCP server.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"
MISTRAL_MODEL = "mistral-embed"
EMBEDDING_DIMENSIONS = 1024


class _LRUCache:
    """Thread-safe LRU cache for embedding vectors."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: List[float]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class MistralEmbedder:
    """Embedding client backed by the Mistral API with LRU caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_size: int = 1000,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self._timeout = timeout
        self._cache = _LRUCache(maxsize=cache_size)

    @property
    def api_key(self) -> str:
        return self._api_key

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single text string, using cache when available."""
        if not text or not text.strip():
            return None

        cache_key = text.strip()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = self._call_api([text])
            if result and len(result) == 1:
                self._cache.put(cache_key, result[0])
                return result[0]
            return None
        except Exception as e:
            logger.debug("Mistral embed failed for text (len=%d): %s", len(text), e)
            return None

    def _call_api(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call the Mistral embeddings API with a list of texts."""
        if not self._api_key:
            logger.debug("Mistral API key not configured")
            return None

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MISTRAL_MODEL,
            "input": texts,
        }

        try:
            resp = requests.post(
                MISTRAL_API_URL,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            embeddings_data = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in embeddings_data]
        except requests.exceptions.Timeout:
            logger.warning("Mistral API timeout after %.1fs", self._timeout)
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning("Mistral API HTTP error: %s", e)
            return None
        except Exception as e:
            logger.warning("Mistral API call failed: %s", e)
            return None

    def is_available(self) -> bool:
        """Check if the embedder has an API key configured."""
        return bool(self._api_key)
