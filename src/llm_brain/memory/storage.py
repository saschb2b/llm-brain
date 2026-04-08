"""Storage operations for memories."""

import json
import time
from datetime import datetime, timezone
from sqlite3 import IntegrityError
from typing import Any, Optional

import numpy as np

from llm_brain.core.config import BrainConfig, get_config
from llm_brain.core.database import Database, get_database
from llm_brain.memory.models import Memory, MemoryTier


class MemoryStorage:
    """CRUD operations for memories."""

    def __init__(
        self, database: Optional[Database] = None, config: Optional[BrainConfig] = None
    ) -> None:
        """Initialize storage.

        Args:
            database: Database instance
            config: Brain configuration
        """
        self.db = database or get_database()
        self.config = config or get_config()

    def memorize(self, memory: Memory) -> str:
        """Store a new memory.

        Args:
            memory: Memory to store

        Returns:
            Memory ID

        Raises:
            IntegrityError: If content_hash already exists
        """
        start_time = time.time()

        # Insert into memories table
        data = memory.to_db_dict()

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        self.db.execute(
            f"INSERT INTO memories ({columns}) VALUES ({placeholders})", tuple(data.values())
        )

        # Insert into vector table if sqlite-vec is available
        self._insert_vector(memory)

        # Insert relations
        self._insert_relations(memory)

        self.db.commit()

        # Log operation
        latency = int((time.time() - start_time) * 1000)
        self._log_operation("store", memory.memory_id, None, latency)

        return memory.memory_id

    def _insert_vector(self, memory: Memory) -> None:
        """Insert vector into sqlite-vec table."""
        if memory.embedding is None:
            return

        try:
            # Convert vector to JSON array for sqlite-vec
            vector_json = json.dumps(memory.embedding.vector.tolist())
            self.db.execute(
                "INSERT INTO vec_memories (memory_id, embedding) VALUES (?, ?)",
                (memory.memory_id, vector_json),
            )
        except Exception:
            # sqlite-vec might not be available or table not created
            pass

    def _insert_relations(self, memory: Memory) -> None:
        """Insert relations into database."""
        if not memory.relations:
            return

        relations_data = [
            (
                memory.memory_id,
                rel.target_id,
                rel.relation_type,
                rel.weight,
                int(datetime.now(timezone.utc).timestamp()),
            )
            for rel in memory.relations
        ]

        self.db.executemany(
            """
            INSERT OR REPLACE INTO relations
            (source_id, target_id, relation_type, weight, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            relations_data,
        )

    def recall(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory if found, None otherwise
        """
        start_time = time.time()

        cursor = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()

        if row is None:
            latency = int((time.time() - start_time) * 1000)
            self._log_operation("retrieve", None, f"id:{memory_id}", latency)
            return None

        # Update access stats
        self._touch_memory(memory_id)

        # Load relations
        memory = Memory.from_db_row(row, self.config.vector_dimensions)
        memory.relations = self._load_relations(memory_id)

        latency = int((time.time() - start_time) * 1000)
        self._log_operation("retrieve", memory_id, f"id:{memory_id}", latency)

        return memory

    def _touch_memory(self, memory_id: str) -> None:
        """Update access timestamp and count."""
        now = int(datetime.now(timezone.utc).timestamp())
        self.db.execute(
            """
            UPDATE memories
            SET accessed_at = ?, access_count = access_count + 1
            WHERE id = ?
            """,
            (now, memory_id),
        )
        self.db.commit()

    def _load_relations(self, memory_id: str) -> list[Any]:
        """Load relations for a memory."""
        from llm_brain.memory.models import Relation

        cursor = self.db.execute("SELECT * FROM relations WHERE source_id = ?", (memory_id,))

        return [
            Relation(
                target_id=row["target_id"], relation_type=row["relation_type"], weight=row["weight"]
            )
            for row in cursor.fetchall()
        ]

    def recall_by_similarity(
        self, query_vector: np.ndarray, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[tuple[Memory, float]]:
        """Retrieve memories by vector similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            tier: Filter by memory tier

        Returns:
            List of (memory, similarity_score) tuples
        """
        start_time = time.time()

        # Try sqlite-vec first if available
        results = self._search_vec_table(query_vector, top_k, tier)

        if not results:
            # Fallback: brute force search
            results = self._brute_force_search(query_vector, top_k, tier)

        latency = int((time.time() - start_time) * 1000)
        context = f"vector_similarity:tier={tier.value if tier else 'all'}"
        self._log_operation("retrieve", None, context, latency)

        return results

    def _search_vec_table(
        self, query_vector: np.ndarray, top_k: int, tier: Optional[MemoryTier]
    ) -> list[tuple[Memory, float]]:
        """Search using sqlite-vec virtual table."""
        results: list[tuple[Memory, float]] = []

        try:
            # Check if vec_memories table exists
            cursor = self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
            )
            if not cursor.fetchone():
                return []

            vector_json = json.dumps(query_vector.tolist())

            if tier:
                # Join with memories table to filter by tier
                cursor = self.db.execute(
                    """
                    SELECT v.memory_id, v.distance, m.*
                    FROM vec_memories v
                    JOIN memories m ON v.memory_id = m.id
                    WHERE m.tier = ?
                    ORDER BY v.distance
                    LIMIT ?
                    """,
                    (tier.value, top_k),
                )
            else:
                cursor = self.db.execute(
                    """
                    SELECT v.memory_id, v.distance, m.*
                    FROM vec_memories v
                    JOIN memories m ON v.memory_id = m.id
                    ORDER BY v.distance
                    LIMIT ?
                    """,
                    (top_k,),
                )

            for row in cursor.fetchall():
                memory = Memory.from_db_row(row, self.config.vector_dimensions)
                # Convert distance to similarity (sqlite-vec returns distance)
                similarity = 1.0 / (1.0 + row["distance"])
                results.append((memory, similarity))

        except Exception:
            pass

        return results

    def _brute_force_search(
        self, query_vector: np.ndarray, top_k: int, tier: Optional[MemoryTier]
    ) -> list[tuple[Memory, float]]:
        """Brute force cosine similarity search."""
        from llm_brain.core.database import blob_to_vector

        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        if tier:
            cursor = self.db.execute(
                "SELECT * FROM memories WHERE tier = ? AND vector IS NOT NULL", (tier.value,)
            )
        else:
            cursor = self.db.execute("SELECT * FROM memories WHERE vector IS NOT NULL")

        scores: list[tuple[Memory, float]] = []

        for row in cursor.fetchall():
            if row["vector"] is None:
                continue

            vector = blob_to_vector(row["vector"], self.config.vector_dimensions)
            vector_norm = vector / (np.linalg.norm(vector) + 1e-8)

            # Cosine similarity
            similarity = float(np.dot(query_norm, vector_norm))

            memory = Memory.from_db_row(row, self.config.vector_dimensions)
            scores.append((memory, similarity))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def recall_by_importance(self, tier: MemoryTier, top_k: int = 5) -> list[Memory]:
        """Retrieve most important memories from a tier.

        Args:
            tier: Memory tier to search
            top_k: Number of results

        Returns:
            List of memories sorted by importance
        """
        cursor = self.db.execute(
            """
            SELECT * FROM memories
            WHERE tier = ?
            ORDER BY importance DESC, access_count DESC
            LIMIT ?
            """,
            (tier.value, top_k),
        )

        return [Memory.from_db_row(row, self.config.vector_dimensions) for row in cursor.fetchall()]

    def update_tier(self, memory_id: str, new_tier: MemoryTier) -> bool:
        """Update memory tier.

        Args:
            memory_id: Memory to update
            new_tier: New tier value

        Returns:
            True if updated, False if not found
        """
        cursor = self.db.execute(
            "UPDATE memories SET tier = ? WHERE id = ?", (new_tier.value, memory_id)
        )
        self.db.commit()

        if cursor.rowcount > 0:
            op = "promote" if self._is_promotion(new_tier) else "demote"
            self._log_operation(op, memory_id, f"to:{new_tier.value}", 0)
            return True
        return False

    def _is_promotion(self, tier: MemoryTier) -> bool:
        """Check if tier change is a promotion."""
        order = {MemoryTier.WORKING: 0, MemoryTier.EPISODIC: 1, MemoryTier.SEMANTIC: 2}
        return order.get(tier, 0) > 0

    def forget(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory to delete

        Returns:
            True if deleted, False if not found
        """
        # Delete from vector table first
        try:
            self.db.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
        except Exception:
            pass

        # Delete relations
        self.db.execute(
            "DELETE FROM relations WHERE source_id = ? OR target_id = ?", (memory_id, memory_id)
        )

        # Delete memory
        cursor = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()

        if cursor.rowcount > 0:
            self._log_operation("forget", memory_id, None, 0)
            return True
        return False

    def _log_operation(
        self, operation: str, memory_id: Optional[str], context: Optional[str], latency_ms: int
    ) -> None:
        """Log operation to cognition_log."""
        try:
            now = int(datetime.now(timezone.utc).timestamp())
            self.db.execute(
                """
                INSERT INTO cognition_log (timestamp, operation, memory_id, context_query, latency_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, operation, memory_id, context, latency_ms),
            )
            self.db.commit()
        except Exception:
            # Don't fail on logging errors
            pass
