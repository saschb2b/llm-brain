"""Data models for memories and related entities."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryTier(str, Enum):
    """Memory hierarchy tiers."""

    WORKING = "working"  # Hot, fast access (cache-like)
    EPISODIC = "episodic"  # Time-indexed interaction history
    SEMANTIC = "semantic"  # Long-term abstracted knowledge


class Provenance(BaseModel):
    """Source information for a memory."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Source type: user_query, web_search, code_execution")
    session_id: Optional[str] = None
    trace: Optional[str] = Field(None, description="Cognitive chain ID")


class Embedding(BaseModel):
    """Embedding vector with metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(default="text-embedding-3-large")
    vector: np.ndarray = Field(..., description="Embedding vector")
    dimensions: int = Field(default=3072)

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: Any) -> np.ndarray:
        """Ensure vector is 1D numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        if v.ndim != 1:
            v = v.flatten()
        result: np.ndarray = v.astype(np.float32)
        return result

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int, info: Any) -> int:
        """Ensure dimensions match vector length."""
        if "vector" in info.data and len(info.data["vector"]) != v:
            return len(info.data["vector"])
        return v


class Relation(BaseModel):
    """Relationship between memories."""

    model_config = ConfigDict(frozen=True)

    target_id: str
    relation_type: str
    weight: float = Field(ge=0.0, le=1.0, default=1.0)

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, v: str) -> str:
        """Normalize relation type."""
        return v.lower().strip()


class MemoryMetadata(BaseModel):
    """Metadata for memory tracking."""

    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    compression_ratio: float = Field(ge=0.0, le=1.0, default=1.0)
    access_count: int = Field(ge=0, default=0)
    decay_rate: float = Field(ge=0.0, default=0.01)

    def calculate_effective_importance(self, days_since_access: float) -> float:
        """Calculate importance adjusted for time decay.

        Args:
            days_since_access: Days since last access

        Returns:
            Decay-adjusted importance score
        """
        decay: float = float(np.exp(-self.decay_rate * days_since_access))
        return self.importance_score * decay


class Memory(BaseModel):
    """Core memory entity.

    This is the primary data structure for storing LLM-native memories.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core identifiers
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tier: MemoryTier = Field(default=MemoryTier.WORKING)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Content
    embedding: Optional[Embedding] = None
    content_hash: Optional[str] = None
    raw_text: Optional[str] = None  # Debug/human-readable only

    # Token mapping
    source_tokens: Optional[list[int]] = None
    context_window_id: Optional[str] = None

    # Metadata
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)

    # Relations
    relations: list[Relation] = Field(default_factory=list)

    # Provenance
    provenance: Optional[Provenance] = None

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage.

        Returns:
            Dictionary suitable for SQLite insertion
        """
        import json

        vector_blob: Optional[bytes] = None
        if self.embedding is not None:
            vector_blob = self.embedding.vector.astype(np.float16).tobytes()

        token_map: Optional[str] = None
        if self.source_tokens or self.context_window_id:
            token_map = json.dumps(
                {"source_tokens": self.source_tokens, "context_window_id": self.context_window_id}
            )

        return {
            "id": self.memory_id,
            "tier": self.tier.value,
            "created_at": int(self.created_at.timestamp()),
            "accessed_at": int(self.last_accessed.timestamp()),
            "vector": vector_blob,
            "content_hash": self.content_hash,
            "importance": self.metadata.importance_score,
            "confidence": self.metadata.confidence,
            "compression_ratio": self.metadata.compression_ratio,
            "access_count": self.metadata.access_count,
            "decay_rate": self.metadata.decay_rate,
            "metadata": json.dumps(
                self.metadata.model_dump(
                    exclude={
                        "importance_score",
                        "confidence",
                        "compression_ratio",
                        "access_count",
                        "decay_rate",
                    }
                )
            ),
            "raw_text": self.raw_text,
            "token_map": token_map,
            "context_window_id": self.context_window_id,
        }

    @classmethod
    def from_db_row(cls, row: Any, dimensions: int = 3072) -> "Memory":
        """Create Memory from database row.

        Args:
            row: SQLite row
            dimensions: Expected vector dimensions

        Returns:
            Memory instance
        """
        import json

        # Parse embedding
        embedding: Optional[Embedding] = None
        if row["vector"]:
            vector = np.frombuffer(row["vector"], dtype=np.float16, count=dimensions).astype(
                np.float32
            )
            embedding = Embedding(vector=vector, dimensions=len(vector))

        # Parse metadata
        meta_dict: dict[str, Any] = {}
        if row["metadata"]:
            try:
                meta_dict = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        # Parse token map
        source_tokens: Optional[list[int]] = None
        context_window_id: Optional[str] = None
        if row["token_map"]:
            try:
                token_map = json.loads(row["token_map"])
                source_tokens = token_map.get("source_tokens")
                context_window_id = token_map.get("context_window_id")
            except json.JSONDecodeError:
                pass

        return cls(
            memory_id=row["id"],
            tier=MemoryTier(row["tier"]),
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
            last_accessed=datetime.fromtimestamp(row["accessed_at"], tz=timezone.utc),
            embedding=embedding,
            content_hash=row["content_hash"],
            raw_text=row["raw_text"],
            source_tokens=source_tokens,
            context_window_id=context_window_id or row["context_window_id"],
            metadata=MemoryMetadata(
                importance_score=row["importance"],
                confidence=row["confidence"],
                compression_ratio=row["compression_ratio"],
                access_count=row["access_count"],
                decay_rate=row["decay_rate"],
                **meta_dict,
            ),
        )

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now(timezone.utc)
        self.metadata.access_count += 1

    def add_relation(self, target_id: str, relation_type: str, weight: float = 1.0) -> None:
        """Add a relation to another memory.

        Args:
            target_id: Target memory ID
            relation_type: Type of relation (e.g., "contradicts", "extends", "causes")
            weight: Relation weight (0.0-1.0)
        """
        relation = Relation(target_id=target_id, relation_type=relation_type, weight=weight)
        self.relations.append(relation)
