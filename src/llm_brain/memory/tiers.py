"""Memory tier management and promotion/demotion logic."""

from typing import Optional

from llm_brain.core.config import BrainConfig, get_config
from llm_brain.memory.models import MemoryTier
from llm_brain.memory.storage import MemoryStorage


class TierManager:
    """Manages memory movement between tiers."""

    def __init__(
        self, storage: Optional[MemoryStorage] = None, config: Optional[BrainConfig] = None
    ) -> None:
        """Initialize tier manager.

        Args:
            storage: Memory storage instance
            config: Brain configuration
        """
        self.storage = storage or MemoryStorage()
        self.config = config or get_config()

    def promote(self, memory_id: str) -> bool:
        """Promote memory to next tier (working -> episodic -> semantic).

        Args:
            memory_id: Memory to promote

        Returns:
            True if promoted, False if already at top tier or not found
        """
        memory = self.storage.recall(memory_id)
        if memory is None:
            return False

        new_tier = self._get_next_tier(memory.tier)
        if new_tier is None:
            return False

        return self.storage.update_tier(memory_id, new_tier)

    def demote(self, memory_id: str) -> bool:
        """Demote memory to previous tier.

        Args:
            memory_id: Memory to demote

        Returns:
            True if demoted, False if already at bottom tier or not found
        """
        memory = self.storage.recall(memory_id)
        if memory is None:
            return False

        new_tier = self._get_prev_tier(memory.tier)
        if new_tier is None:
            return False

        return self.storage.update_tier(memory_id, new_tier)

    def _get_next_tier(self, tier: MemoryTier) -> Optional[MemoryTier]:
        """Get next tier in hierarchy."""
        progression = {
            MemoryTier.WORKING: MemoryTier.EPISODIC,
            MemoryTier.EPISODIC: MemoryTier.SEMANTIC,
            MemoryTier.SEMANTIC: None,
        }
        return progression.get(tier)

    def _get_prev_tier(self, tier: MemoryTier) -> Optional[MemoryTier]:
        """Get previous tier in hierarchy."""
        regression = {
            MemoryTier.SEMANTIC: MemoryTier.EPISODIC,
            MemoryTier.EPISODIC: MemoryTier.WORKING,
            MemoryTier.WORKING: None,
        }
        return regression.get(tier)

    def auto_promote_by_importance(self) -> int:
        """Auto-promote memories based on importance thresholds.

        Returns:
            Number of memories promoted
        """
        promoted = 0

        # Promote high-importance working memories to episodic
        working_memories = self.storage.recall_by_importance(
            MemoryTier.WORKING, top_k=self.config.working_memory_limit
        )

        for memory in working_memories:
            if memory.metadata.importance_score >= self.config.importance_threshold:
                if self.promote(memory.memory_id):
                    promoted += 1

        # Promote high-importance episodic memories to semantic
        episodic_memories = self.storage.recall_by_importance(
            MemoryTier.EPISODIC, top_k=self.config.episodic_memory_limit
        )

        for memory in episodic_memories:
            if memory.metadata.importance_score >= self.config.importance_threshold:
                if self.promote(memory.memory_id):
                    promoted += 1

        return promoted

    def evict_lru(self, tier: MemoryTier, target_count: Optional[int] = None) -> int:
        """Evict least recently used memories from a tier.

        Args:
            tier: Tier to evict from
            target_count: Target memory count (uses config limit if None)

        Returns:
            Number of memories evicted
        """
        if tier == MemoryTier.WORKING:
            limit = target_count or self.config.working_memory_limit
        elif tier == MemoryTier.EPISODIC:
            limit = target_count or self.config.episodic_memory_limit
        else:
            # Don't auto-evict from semantic
            return 0

        # Get current count
        cursor = self.storage.db.execute(
            "SELECT COUNT(*) as count FROM memories WHERE tier = ?", (tier.value,)
        )
        row = cursor.fetchone()
        current_count = row["count"] if row else 0

        if current_count <= limit:
            return 0

        # Find memories to evict (least accessed, lowest importance)
        to_evict = current_count - limit

        cursor = self.storage.db.execute(
            """
            SELECT id FROM memories
            WHERE tier = ?
            ORDER BY accessed_at ASC, importance ASC
            LIMIT ?
            """,
            (tier.value, to_evict),
        )

        evicted = 0
        for row in cursor.fetchall():
            if tier == MemoryTier.WORKING:
                # Demote to episodic instead of deleting
                self.storage.update_tier(row["id"], MemoryTier.EPISODIC)
            else:
                # Delete from episodic
                self.storage.forget(row["id"])
            evicted += 1

        return evicted

    def compress_memory(self, memory_id: str) -> bool:
        """Mark memory as compressed (episodic -> semantic transformation).

        This is a placeholder for actual compression logic which would
        involve LLM-based summarization.

        Args:
            memory_id: Memory to compress

        Returns:
            True if compressed, False if not found
        """
        memory = self.storage.recall(memory_id)
        if memory is None:
            return False

        # Update compression ratio
        self.storage.db.execute(
            """
            UPDATE memories
            SET compression_ratio = 0.3, tier = ?
            WHERE id = ?
            """,
            (MemoryTier.SEMANTIC.value, memory_id),
        )
        self.storage.db.commit()

        self.storage._log_operation("compress", memory_id, "ratio:0.3", 0)
        return True

    def consolidate(self) -> dict[str, int]:
        """Run full memory consolidation cycle.

        This includes:
        - Auto-promotion by importance
        - LRU eviction from working memory
        - LRU eviction from episodic memory

        Returns:
            Dictionary with counts of operations performed
        """
        results = {"promoted": 0, "working_evicted": 0, "episodic_evicted": 0, "compressed": 0}

        # Auto-promote important memories
        results["promoted"] = self.auto_promote_by_importance()

        # Evict from working memory
        results["working_evicted"] = self.evict_lru(MemoryTier.WORKING)

        # Evict from episodic memory
        results["episodic_evicted"] = self.evict_lru(MemoryTier.EPISODIC)

        return results

    def get_tier_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics for each tier.

        Returns:
            Dictionary with tier statistics
        """
        stats = {}

        for tier in MemoryTier:
            cursor = self.storage.db.execute(
                """
                SELECT
                    COUNT(*) as count,
                    AVG(importance) as avg_importance,
                    AVG(access_count) as avg_accesses,
                    MAX(accessed_at) as last_access
                FROM memories
                WHERE tier = ?
                """,
                (tier.value,),
            )
            row = cursor.fetchone()

            stats[tier.value] = {
                "count": row["count"] or 0,
                "avg_importance": round(row["avg_importance"] or 0, 3),
                "avg_accesses": round(row["avg_accesses"] or 0, 1),
                "last_access": row["last_access"],
            }

        return stats
