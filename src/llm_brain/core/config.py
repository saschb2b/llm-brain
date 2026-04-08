"""Configuration management for LLM-Brain."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class BrainConfig:
    """Configuration for LLM-Brain.

    Attributes:
        brain_path: Directory path for brain storage
        db_name: SQLite database filename
        vector_dimensions: Embedding vector dimensions
        default_embedding_model: Default embedding model name
        working_memory_limit: Max items in working memory tier
        episodic_memory_limit: Max items in episodic memory tier
        importance_threshold: Threshold for auto-promotion to semantic
    """

    brain_path: Union[Path, str] = field(default_factory=lambda: Path.home() / ".llm-brain")
    db_name: str = "core.db"
    vector_dimensions: int = 3072
    default_embedding_model: str = "text-embedding-3-large"
    working_memory_limit: int = 1000
    episodic_memory_limit: int = 10000
    importance_threshold: float = 0.8
    decay_rate: float = 0.01

    def __post_init__(self) -> None:
        """Ensure brain_path is a Path object."""
        if isinstance(self.brain_path, str):
            self.brain_path = Path(self.brain_path)

    @property
    def db_path(self) -> Path:
        """Get full path to SQLite database."""
        return self.brain_path / self.db_name  # type: ignore[operator]

    @property
    def kuzu_path(self) -> Path:
        """Get path to KùzuDB graph database."""
        return self.brain_path / "graph.kuzu"  # type: ignore[operator]

    @property
    def log_path(self) -> Path:
        """Get path to cognition log."""
        return self.brain_path / "cognition.jsonl"  # type: ignore[operator]

    def ensure_directories(self) -> None:
        """Create brain directory structure if not exists."""
        self.brain_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]


# Global config instance
_config: Optional[BrainConfig] = None


def get_config(
    brain_path: Optional[str] = None,
    vector_dimensions: Optional[int] = None,
) -> BrainConfig:
    """Get or create global configuration.

    Args:
        brain_path: Override brain storage path
        vector_dimensions: Override vector dimensions

    Returns:
        BrainConfig instance
    """
    global _config  # noqa: PLW0603

    if _config is None:
        # Check environment variables
        env_path = os.environ.get("LLM_BRAIN_PATH")
        env_dims = os.environ.get("LLM_BRAIN_DIMENSIONS")

        path = brain_path or env_path
        dims = vector_dimensions or (int(env_dims) if env_dims else None)

        kwargs: dict[str, Any] = {}
        if path:
            kwargs["brain_path"] = Path(path)
        if dims:
            kwargs["vector_dimensions"] = dims

        _config = BrainConfig(**kwargs)
        _config.ensure_directories()

    return _config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config  # noqa: PLW0603
    _config = None
