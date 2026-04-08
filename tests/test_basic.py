"""Basic functionality tests."""

import tempfile

import numpy as np
import pytest

from llm_brain import Brain, MemoryTier
from llm_brain.core.config import reset_config
from llm_brain.core.database import reset_database


@pytest.fixture
def temp_brain():
    """Create a temporary brain for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reset_config()
        reset_database()

        brain = Brain(brain_path=tmpdir, vector_dimensions=128, auto_init=True)
        yield brain

        brain.close()
        reset_config()
        reset_database()


class TestInitialization:
    """Test brain initialization."""

    def test_initialization_creates_database(self, temp_brain):
        """Test that initialization creates the database."""
        assert temp_brain.db.is_initialized()
        assert temp_brain.config.db_path.exists()

    def test_schema_version_set(self, temp_brain):
        """Test that schema version is set."""
        version = temp_brain.db.get_schema_version()
        assert version == "1.0"

    def test_health_check_passes(self, temp_brain):
        """Test health check passes for initialized brain."""
        health = temp_brain.health_check()
        assert health["initialized"]
        assert health["writable"]


class TestMemoryStorage:
    """Test memory storage operations."""

    def test_memorize_and_recall(self, temp_brain):
        """Test basic memorize and recall."""
        vector = np.random.randn(128).astype(np.float32)

        memory_id = temp_brain.memorize(vector=vector, text="Test memory", importance=0.8)

        assert memory_id is not None

        # Recall by ID
        memory = temp_brain.recall(memory_id=memory_id)
        assert memory is not None
        assert memory.memory_id == memory_id
        assert memory.raw_text == "Test memory"
        assert memory.metadata.importance_score == 0.8

    def test_recall_by_similarity(self, temp_brain):
        """Test similarity search."""
        # Store a memory
        vector = np.ones(128).astype(np.float32)
        temp_brain.memorize(vector=vector, text="Ones vector")

        # Search with similar vector
        query = np.ones(128).astype(np.float32) * 0.9
        results = temp_brain.recall(query_vector=query, top_k=5)

        assert len(results) > 0
        assert results[0][0].raw_text == "Ones vector"
        assert results[0][1] > 0.5  # Similarity score

    def test_recall_by_importance(self, temp_brain):
        """Test importance-based recall."""
        # Store memories with different importance
        for i in range(5):
            temp_brain.memorize(vector=np.random.randn(128).astype(np.float32), importance=0.1 * i)

        # Get most important
        important = temp_brain.recall_important(MemoryTier.WORKING, top_k=3)

        assert len(important) == 3
        # Should be sorted by importance descending
        assert important[0].metadata.importance_score >= important[1].metadata.importance_score

    def test_forget(self, temp_brain):
        """Test memory deletion."""
        vector = np.random.randn(128).astype(np.float32)
        memory_id = temp_brain.memorize(vector=vector)

        # Verify exists
        assert temp_brain.recall(memory_id=memory_id) is not None

        # Delete
        assert temp_brain.forget(memory_id)

        # Verify deleted
        assert temp_brain.recall(memory_id=memory_id) is None

    def test_duplicate_detection(self, temp_brain):
        """Test that similar memories have different IDs."""
        vector = np.random.randn(128).astype(np.float32)

        id1 = temp_brain.memorize(vector=vector, text="Same")
        id2 = temp_brain.memorize(vector=vector, text="Same")

        # Different IDs (we don't deduplicate automatically)
        assert id1 != id2


class TestRelations:
    """Test memory relations."""

    def test_add_relation(self, temp_brain):
        """Test adding relations between memories."""
        # Create two memories
        id1 = temp_brain.memorize(vector=np.random.randn(128).astype(np.float32), text="Memory 1")
        id2 = temp_brain.memorize(vector=np.random.randn(128).astype(np.float32), text="Memory 2")

        # Add relation
        assert temp_brain.relate(id1, id2, "extends", 0.8)

        # Verify relation
        memory = temp_brain.recall(memory_id=id1)
        assert len(memory.relations) == 1
        assert memory.relations[0].target_id == id2
        assert memory.relations[0].relation_type == "extends"

    def test_recall_related(self, temp_brain):
        """Test recalling related memories."""
        id1 = temp_brain.memorize(vector=np.random.randn(128).astype(np.float32), relations=[])
        id2 = temp_brain.memorize(
            vector=np.random.randn(128).astype(np.float32), relations=[(id1, "causes", 0.9)]
        )

        # Get related
        related = temp_brain.recall_related(id2)

        assert len(related) == 1
        assert related[0]["memory_id"] == id1


class TestTiers:
    """Test memory tier management."""

    def test_promote_memory(self, temp_brain):
        """Test memory promotion."""
        # Create working memory
        memory_id = temp_brain.memorize(
            vector=np.random.randn(128).astype(np.float32), tier=MemoryTier.WORKING, importance=0.9
        )

        memory = temp_brain.recall(memory_id=memory_id)
        assert memory.tier == MemoryTier.WORKING

        # Promote
        assert temp_brain.tier_manager.promote(memory_id)

        # Verify
        memory = temp_brain.recall(memory_id=memory_id)
        assert memory.tier == MemoryTier.EPISODIC

    def test_auto_promote(self, temp_brain):
        """Test automatic promotion by importance."""
        # Create high-importance memory
        temp_brain.memorize(
            vector=np.random.randn(128).astype(np.float32),
            tier=MemoryTier.WORKING,
            importance=0.95,  # Above threshold
        )

        # Run auto-promote
        count = temp_brain.tier_manager.auto_promote_by_importance()

        assert count >= 1

    def test_tier_stats(self, temp_brain):
        """Test tier statistics."""
        # Add memories to different tiers
        temp_brain.memorize(vector=np.random.randn(128).astype(np.float32), tier=MemoryTier.WORKING)
        temp_brain.memorize(
            vector=np.random.randn(128).astype(np.float32), tier=MemoryTier.EPISODIC
        )

        stats = temp_brain.tier_manager.get_tier_stats()

        assert stats["working"]["count"] >= 1
        assert stats["episodic"]["count"] >= 1


class TestStats:
    """Test statistics and health check."""

    def test_stats_structure(self, temp_brain):
        """Test stats return correct structure."""
        stats = temp_brain.stats()

        assert "tiers" in stats
        assert "graph" in stats
        assert "logs" in stats
        assert "config" in stats

    def test_health_check_structure(self, temp_brain):
        """Test health check returns correct structure."""
        health = temp_brain.health_check()

        assert "initialized" in health
        assert "writable" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
