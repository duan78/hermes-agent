"""Memory-Claw memory plugin — MemoryProvider for LanceDB-backed semantic memory.

Provides persistent vector memory with LanceDB storage and Mistral embeddings.
Exposes 5 tools (mclaw_store, mclaw_recall, mclaw_forget, mclaw_stats, mclaw_update)
through the MemoryProvider interface.  Includes auto-extraction via LLM.

Storage: LanceDB at $HERMES_HOME/memory-claw/
Embeddings: Mistral API (mistral-embed, 1024 dims)
Extraction: Mistral-small-latest
"""

from __future__ import annotations

from .provider import MemoryClawProvider

__all__ = ["MemoryClawProvider"]


def register(ctx) -> None:
    """Register memory-claw as a memory provider plugin."""
    ctx.register_memory_provider(MemoryClawProvider())
