"""Hook system for automatic brain integration.

This module provides decorators and utilities for automatically
using the brain without manual intervention.

Example:
    from llm_brain.hooks import with_brain, auto_store, auto_recall

    @with_brain
    def my_chat_session():
        brain = get_current_brain()
        # Brain is automatically initialized
        # Memories are loaded at start

    @auto_store(importance=0.8)
    def process_important_fact(fact: str) -> str:
        return f"Processed: {fact}"
        # Automatically stored in brain
"""

from llm_brain.hooks.decorators import (
    auto_recall,
    auto_store,
    get_current_brain,
    with_brain,
)

__all__ = [
    "auto_recall",
    "auto_store",
    "get_current_brain",
    "with_brain",
]
