"""Main API interface for LLM-Brain.

This module provides the primary interface for storing and retrieving
memories with a simple, intuitive API designed for LLM use.
"""

import time
from typing import Any, Optional, Union

import numpy as np

from llm_brain.core.config import BrainConfig, get_config
from llm_brain.core.database import Database, get_database, reset_database
from llm_brain.graph.kuzu_graph import KuzuGraph, get_graph, reset_graph
from llm_brain.memory.models import (
    Embedding,
    Memory,
    MemoryMetadata,
    MemoryTier,
    Provenance,
    Relation,
)
from llm_brain.memory.storage import MemoryStorage
from llm_brain.memory.tiers import TierManager
from llm_brain.utils.hashing import content_hash
from llm_brain.utils.logging import CognitionLogger


class Brain:
    """Main interface for LLM-Brain memory system.

    Example:
        >>> brain = Brain()
        >>>
        >>> # Store a memory
        >>> embedding = Embedding(vector=np.random.randn(3072))
        >>> memory = Memory(
        ...     embedding=embedding,
        ...     raw_text="Important insight",
        ...     metadata=MemoryMetadata(importance_score=0.9)
        ... )
        >>> brain.memorize(memory)

        >>> # Recall by similarity
        >>> query = Embedding(vector=np.random.randn(3072))
        >>> results = brain.recall(query_vector=query.vector, top_k=5)

        >>> # Get important memories
        >>> important = brain.recall_important(MemoryTier.SEMANTIC, top_k=10)
    """

    def __init__(
        self,
        brain_path: Optional[str] = None,
        vector_dimensions: Optional[int] = None,
        auto_init: bool = True,
    ) -> None:
        """Initialize Brain.

        Args:
            brain_path: Path to brain storage directory
            vector_dimensions: Vector dimensions for embeddings
            auto_init: Auto-initialize database if needed
        """
        # Initialize configuration
        self.config = get_config(brain_path, vector_dimensions)

        # Initialize components
        self.db = get_database(self.config)
        self.storage = MemoryStorage(self.db, self.config)
        self.tier_manager = TierManager(self.storage, self.config)
        self.graph = get_graph(self.config)
        self.logger = CognitionLogger(self.config)

        # Initialize schema if needed
        if auto_init and not self.db.is_initialized():
            self.initialize()

    def initialize(self) -> None:
        """Initialize brain storage."""
        self.config.ensure_directories()
        self.db.initialize_schema()

        # Log initialization
        self.logger.log(
            "initialize",
            metadata={
                "schema_version": "1.0",
                "vector_dimensions": self.config.vector_dimensions,
                "db_path": str(self.config.db_path),
            },
        )

    def memorize(
        self,
        memory: Optional[Memory] = None,
        *,
        vector: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        tier: Union[MemoryTier, str] = MemoryTier.WORKING,
        importance: float = 0.5,
        confidence: float = 0.5,
        relations: Optional[list[tuple[str, str, float]]] = None,
        provenance: Optional[Provenance] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a new memory.

        This is the primary method for adding memories. It accepts either
        a Memory object or individual components.

        Args:
            memory: Complete Memory object (if provided, other args ignored)
            vector: Embedding vector
            text: Optional human-readable text
            tier: Memory tier
            importance: Importance score (0.0-1.0)
            confidence: Confidence score (0.0-1.0)
            relations: List of (target_id, relation_type, weight) tuples
            provenance: Source information
            metadata: Additional metadata

        Returns:
            Memory ID

        Example:
            >>> brain = Brain()
            >>> vector = np.random.randn(3072)
            >>> brain.memorize(
            ...     vector=vector,
            ...     text="User prefers Python",
            ...     importance=0.9,
            ...     tier=MemoryTier.WORKING
            ... )
        """
        if memory is None:
            # Build Memory from components
            if vector is None:
                raise ValueError("Either memory or vector must be provided")

            if isinstance(tier, str):
                tier = MemoryTier(tier)

            # Create embedding
            embedding = Embedding(
                vector=vector, dimensions=len(vector), model=self.config.default_embedding_model
            )

            # Create metadata
            meta = MemoryMetadata(importance_score=importance, confidence=confidence)

            # Create content hash
            hash_val = content_hash(text=text, vector=vector, metadata=metadata)

            # Build relations
            rels: list[Relation] = []
            if relations:
                rels = [
                    Relation(target_id=tid, relation_type=rtype, weight=w)
                    for tid, rtype, w in relations
                ]

            memory = Memory(
                tier=tier,
                embedding=embedding,
                content_hash=hash_val,
                raw_text=text,
                metadata=meta,
                relations=rels,
                provenance=provenance,
            )

        # Store in database
        memory_id = self.storage.memorize(memory)

        # Also store in graph if available
        if self.graph.is_available and memory.embedding:
            self.graph.add_memory_node(
                memory_id, memory.tier.value, memory.metadata.importance_score
            )
            for rel in memory.relations:
                self.graph.add_relation(memory_id, rel.target_id, rel.relation_type, rel.weight)

        return memory_id

    def recall(
        self,
        memory_id: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5,
        tier: Union[MemoryTier, str, None] = None,
    ) -> Union[Optional[Memory], list[tuple[Memory, float]]]:
        """Retrieve memories.

        Supports three modes:
        1. By ID: Returns single Memory or None
        2. By vector similarity: Returns list of (Memory, similarity)

        Args:
            memory_id: Exact memory ID to retrieve
            query_vector: Vector for similarity search
            top_k: Number of results for similarity search
            tier: Filter by memory tier

        Returns:
            Memory, list of (Memory, score), or None

        Example:
            >>> # By ID
            >>> memory = brain.recall("uuid-here")

            >>> # By similarity
            >>> query = np.random.randn(3072)
            >>> results = brain.recall(query_vector=query, top_k=5)
        """
        # Convert tier string to enum
        tier_enum: Optional[MemoryTier] = None
        if isinstance(tier, str):
            tier_enum = MemoryTier(tier)
        elif tier is not None:
            tier_enum = tier

        # By ID
        if memory_id:
            return self.storage.recall(memory_id)

        # By similarity
        if query_vector is not None:
            return self.storage.recall_by_similarity(query_vector, top_k, tier_enum)

        raise ValueError("Either memory_id or query_vector must be provided")

    def recall_important(
        self, tier: Union[MemoryTier, str] = MemoryTier.WORKING, top_k: int = 5
    ) -> list[Memory]:
        """Get most important memories from a tier.

        Args:
            tier: Memory tier
            top_k: Number of results

        Returns:
            List of memories sorted by importance
        """
        if isinstance(tier, str):
            tier = MemoryTier(tier)

        return self.storage.recall_by_importance(tier, top_k)

    def recall_related(
        self,
        memory_id: str,
        relation_type: Optional[str] = None,
        min_weight: float = 0.0,
        include_graph: bool = True,
    ) -> list[dict[str, Any]]:
        """Get memories related to a given memory.

        Args:
            memory_id: Source memory ID
            relation_type: Filter by relation type
            min_weight: Minimum relation weight
            include_graph: Also search graph database

        Returns:
            List of related memory info
        """
        # Get from SQLite
        memory = self.storage.recall(memory_id)
        if memory is None:
            return []

        results: list[dict[str, Any]] = []
        for rel in memory.relations:
            if relation_type and rel.relation_type != relation_type:
                continue
            if rel.weight < min_weight:
                continue

            results.append(
                {
                    "memory_id": rel.target_id,
                    "relation_type": rel.relation_type,
                    "weight": rel.weight,
                    "source": "sqlite",
                }
            )

        # Also query graph if available
        if include_graph and self.graph.is_available:
            graph_results = self.graph.get_related_memories(memory_id, relation_type, min_weight)
            for gr in graph_results:
                if gr["id"] not in [r["memory_id"] for r in results]:
                    results.append(
                        {
                            "memory_id": gr["id"],
                            "relation_type": gr.get("relation_type", "related"),
                            "weight": gr["weight"],
                            "source": "graph",
                        }
                    )

        return results

    def forget(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory to delete

        Returns:
            True if deleted, False if not found
        """
        # Delete from graph first
        if self.graph.is_available:
            self.graph.delete_memory(memory_id)

        return self.storage.forget(memory_id)

    def relate(
        self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0
    ) -> bool:
        """Create a relation between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relation_type: Type of relation (e.g., "contradicts", "extends")
            weight: Relation weight (0.0-1.0)

        Returns:
            True if relation created
        """
        # Get source memory and add relation
        memory = self.storage.recall(source_id)
        if memory is None:
            return False

        memory.add_relation(target_id, relation_type, weight)

        # Update in storage
        self.storage._insert_relations(memory)

        # Also add to graph
        if self.graph.is_available:
            self.graph.add_relation(source_id, target_id, relation_type, weight)

        return True

    def consolidate(self) -> dict[str, int]:
        """Run memory consolidation.

        This includes:
        - Promoting important memories
        - Evicting LRU items from working/episodic
        - Compressing old episodic memories

        Returns:
            Dictionary with operation counts
        """
        return self.tier_manager.consolidate()

    def stats(self) -> dict[str, Any]:
        """Get brain statistics.

        Returns:
            Dictionary with tier stats, graph stats, and log stats
        """
        return {
            "tiers": self.tier_manager.get_tier_stats(),
            "graph": self.graph.get_statistics(),
            "logs": self.logger.get_stats(),
            "config": {
                "brain_path": str(self.config.brain_path),
                "vector_dimensions": self.config.vector_dimensions,
                "working_limit": self.config.working_memory_limit,
                "episodic_limit": self.config.episodic_memory_limit,
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Check brain health.

        Returns:
            Health status dictionary
        """
        status = {
            "initialized": self.db.is_initialized(),
            "schema_version": self.db.get_schema_version(),
            "sqlite_vec_loaded": self.db._vec_loaded,
            "graph_available": self.graph.is_available,
            "writable": False,
        }

        # Test write
        try:
            test_vector = np.random.randn(self.config.vector_dimensions).astype(np.float32)
            test_id = self.memorize(vector=test_vector, text="_health_check_test", importance=0.0)
            self.forget(test_id)
            status["writable"] = True
        except Exception as e:
            status["write_error"] = str(e)

        return status

    def close(self) -> None:
        """Close all connections."""
        self.db.close()
        reset_database()
        reset_graph()

    def __enter__(self) -> "Brain":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


# Convenience functions for IPython-style usage
def create_brain(brain_path: Optional[str] = None, vector_dimensions: int = 3072) -> Brain:
    """Create and initialize a new Brain instance.

    Args:
        brain_path: Storage path (default: ~/.kimi-brain)
        vector_dimensions: Embedding dimensions

    Returns:
        Initialized Brain instance
    """
    brain = Brain(brain_path, vector_dimensions)
    if not brain.db.is_initialized():
        brain.initialize()
    return brain


def bootstrap(brain_path: Optional[str] = None, vector_dimensions: int = 3072) -> dict[str, Any]:
    """Bootstrap a new brain instance and return health status.

    This is the primary entry point for first-time setup.

    Args:
        brain_path: Storage path
        vector_dimensions: Embedding dimensions

    Returns:
        Health status dictionary
    """
    brain = create_brain(brain_path, vector_dimensions)
    health = brain.health_check()
    brain.close()
    return health
