"""Decorators for automatic brain integration.

Usage:
    @with_brain
    def my_function():
        # Automatically loads memories at start
        # Can access brain via get_current_brain()
        pass

    @auto_store(importance=0.8)
    def process_user_input(text: str) -> str:
        # Automatically stores the result
        return f"Processed: {text}"
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import numpy as np

from llm_brain import Brain

F = TypeVar("F", bound=Callable[..., Any])

# Thread-local storage for current brain instance
_current_brain: Brain | None = None


def get_current_brain() -> Brain | None:
    """Get the brain instance for the current context."""
    return _current_brain


def _simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """Simple deterministic embedding for hooks."""
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)


def with_brain(
    brain_path: str | None = None,
    vector_dimensions: int = 128,
    load_memories: bool = True,
) -> Callable[[F], F]:
    """Decorator that provides a brain context.

    Automatically:
    - Creates/connects to brain at session start
    - Loads top important memories into context
    - Makes brain available via get_current_brain()

    Args:
        brain_path: Custom brain path (optional)
        vector_dimensions: Embedding dimensions
        load_memories: Whether to load memories at start
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            global _current_brain  # noqa: PLW0603

            # Create brain connection
            if brain_path:
                brain = Brain(
                    brain_path=brain_path,
                    vector_dimensions=vector_dimensions,
                )
            else:
                brain = Brain(vector_dimensions=vector_dimensions)

            _current_brain = brain

            try:
                # Load important memories at start
                if load_memories:
                    from llm_brain.memory.models import MemoryTier  # noqa: PLC0415

                    memories = brain.recall_important(MemoryTier.WORKING, top_k=5)
                    if memories:
                        print("\n🧠 [Brain] Loaded memories:")
                        for m in memories:
                            print(f"   • {m.raw_text[:60]}...")
                        print()

                # Run the function
                return func(*args, **kwargs)

            finally:
                # Store session summary at end
                _store_session_summary(brain)
                brain.close()
                _current_brain = None

        return wrapper  # type: ignore[return-value]

    return decorator


def auto_store(
    importance: float = 0.7,
    extract_text: Callable[..., str] | None = None,
) -> Callable[[F], F]:
    """Decorator that automatically stores function results in brain.

    Args:
        importance: Importance score for stored memories
        extract_text: Function to extract text from result (optional)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Execute function
            result = func(*args, **kwargs)

            # Try to store in brain
            brain = get_current_brain()
            if brain is not None:
                try:
                    # Extract text to store
                    if extract_text:
                        text = extract_text(result)
                    elif isinstance(result, str):
                        text = result
                    else:
                        text = f"{func.__name__}: {str(result)[:200]}"

                    # Generate embedding and store
                    vector = _simple_embed(text)
                    brain.memorize(
                        vector=vector,
                        text=text,
                        importance=importance,
                    )
                except Exception:  # noqa: S110
                    # Don't fail if storage fails
                    pass

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def auto_recall(query_extractor: Callable[..., str]) -> Callable[[F], F]:
    """Decorator that recalls relevant memories before function execution.

    Args:
        query_extractor: Function to extract search query from arguments
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            brain = get_current_brain()

            if brain is not None:
                try:
                    # Extract query
                    query = query_extractor(*args, **kwargs)

                    # Search for relevant memories
                    vector = _simple_embed(query)
                    results = brain.recall(query_vector=vector, top_k=3)

                    if results:
                        print("\n🧠 [Brain] Recalled context:")
                        for memory, _similarity in results:
                            print(f"   • [{memory.tier.value}] {memory.raw_text[:50]}...")
                        print()
                except Exception:  # noqa: S110
                    pass

            # Execute function
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def _store_session_summary(brain: Brain) -> None:
    """Store a summary of the session at the end."""
    try:
        # Get stats
        stats = brain.stats()
        total_memories = sum(tier["count"] for tier in stats["tiers"].values())

        text = f"Session ended. Total memories: {total_memories}"
        vector = _simple_embed(text)
        brain.memorize(
            vector=vector,
            text=text,
            importance=0.5,  # Low importance for session markers
        )
    except Exception:  # noqa: S110
        pass
