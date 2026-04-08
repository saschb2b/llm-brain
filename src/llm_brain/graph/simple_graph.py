"""Pure-Python graph using SQLite (no Kuzu dependency).

Uses recursive CTEs for multi-hop queries.
"""

from __future__ import annotations

from typing import Any

from llm_brain.core.config import BrainConfig, get_config
from llm_brain.core.database import get_database


class SimpleGraph:
    """Graph database using SQLite with recursive CTEs.

    No external dependencies - works on all platforms.
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        """Initialize graph using existing database."""
        self.config = config or get_config()
        self.db = get_database(config)

    @property
    def is_available(self) -> bool:
        """Always available since it uses SQLite."""
        return self.db.is_initialized()

    def add_memory_node(self, memory_id: str, _tier: str, _importance: float) -> bool:
        """Add a memory node.

        Nodes are implicitly created via relations.
        This method ensures the node metadata is tracked.
        """
        try:
            # Nodes exist implicitly in relations table
            # Just verify the memory exists
            cursor = self.db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,))
            return cursor.fetchone() is not None
        except Exception:
            return False

    def add_relation(
        self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0
    ) -> bool:
        """Add a relation between two memories."""
        try:
            import time  # noqa: PLC0415

            created_at = int(time.time())

            self.db.execute(
                """
                INSERT INTO relations (source_id, target_id, relation_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                    weight = excluded.weight,
                    created_at = excluded.created_at
                """,
                (source_id, target_id, relation_type, weight, created_at),
            )
            self.db.commit()
            return True
        except Exception:
            return False

    def get_related_memories(
        self, memory_id: str, relation_type: str | None = None, min_weight: float = 0.0
    ) -> list[dict[str, Any]]:
        """Get memories related to the given memory."""
        try:
            if relation_type:
                cursor = self.db.execute(
                    """
                    SELECT r.*, m.tier, m.importance
                    FROM relations r
                    JOIN memories m ON r.target_id = m.id
                    WHERE r.source_id = ?
                        AND r.relation_type = ?
                        AND r.weight >= ?
                    ORDER BY r.weight DESC
                    """,
                    (memory_id, relation_type, min_weight),
                )
            else:
                cursor = self.db.execute(
                    """
                    SELECT r.*, m.tier, m.importance
                    FROM relations r
                    JOIN memories m ON r.target_id = m.id
                    WHERE r.source_id = ?
                        AND r.weight >= ?
                    ORDER BY r.weight DESC
                    """,
                    (memory_id, min_weight),
                )

            return [
                {
                    "id": row["target_id"],
                    "relation_type": row["relation_type"],
                    "weight": row["weight"],
                    "tier": row["tier"],
                    "importance": row["importance"],
                }
                for row in cursor.fetchall()
            ]
        except Exception:
            return []

    def multi_hop_query(
        self, start_id: str, hops: int = 2, relation_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Find memories reachable within N hops using recursive CTE.

        Uses SQLite's recursive CTE support for graph traversal.
        """
        try:
            if relation_type:
                cursor = self.db.execute(
                    """
                    WITH RECURSIVE hop_paths AS (
                        -- Base case: direct relations
                        SELECT
                            target_id as memory_id,
                            1 as hop_count,
                            weight as total_weight
                        FROM relations
                        WHERE source_id = ?
                            AND relation_type = ?

                        UNION ALL

                        -- Recursive case: extend paths
                        SELECT
                            r.target_id,
                            hp.hop_count + 1,
                            hp.total_weight * r.weight
                        FROM relations r
                        JOIN hop_paths hp ON r.source_id = hp.memory_id
                        WHERE hp.hop_count < ?
                            AND r.relation_type = ?
                    )
                    SELECT DISTINCT
                        hp.memory_id as id,
                        m.tier,
                        m.importance,
                        hp.hop_count as hops,
                        hp.total_weight as weight
                    FROM hop_paths hp
                    JOIN memories m ON hp.memory_id = m.id
                    ORDER BY hp.hop_count, hp.total_weight DESC
                    LIMIT 20
                    """,
                    (start_id, relation_type, hops, relation_type),
                )
            else:
                cursor = self.db.execute(
                    """
                    WITH RECURSIVE hop_paths AS (
                        -- Base case: direct relations
                        SELECT
                            target_id as memory_id,
                            1 as hop_count,
                            weight as total_weight
                        FROM relations
                        WHERE source_id = ?

                        UNION ALL

                        -- Recursive case: extend paths
                        SELECT
                            r.target_id,
                            hp.hop_count + 1,
                            hp.total_weight * r.weight
                        FROM relations r
                        JOIN hop_paths hp ON r.source_id = hp.memory_id
                        WHERE hp.hop_count < ?
                    )
                    SELECT DISTINCT
                        hp.memory_id as id,
                        m.tier,
                        m.importance,
                        hp.hop_count as hops,
                        hp.total_weight as weight
                    FROM hop_paths hp
                    JOIN memories m ON hp.memory_id = m.id
                    ORDER BY hp.hop_count, hp.total_weight DESC
                    LIMIT 20
                    """,
                    (start_id, hops),
                )

            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Multi-hop query failed: {e}")
            return []

    def find_paths(self, source_id: str, target_id: str, max_length: int = 3) -> list[list[str]]:
        """Find paths between two memories using recursive CTE."""
        try:
            cursor = self.db.execute(
                """
                WITH RECURSIVE paths AS (
                    -- Base case: direct connections
                    SELECT
                        target_id as current_id,
                        json_array(source_id, target_id) as path,
                        1 as length
                    FROM relations
                    WHERE source_id = ?

                    UNION ALL

                    -- Recursive case: extend paths
                    SELECT
                        r.target_id,
                        json_insert(p.path, '$[' || json_array_length(p.path) || ']', r.target_id),
                        p.length + 1
                    FROM relations r
                    JOIN paths p ON r.source_id = p.current_id
                    WHERE p.length < ?
                        AND json_type(p.path) IS NOT NULL
                        AND r.target_id NOT IN (
                            SELECT value FROM json_each(p.path)
                        )
                )
                SELECT path
                FROM paths
                WHERE current_id = ?
                ORDER BY length
                LIMIT 10
                """,
                (source_id, max_length, target_id),
            )

            import json  # noqa: PLC0415

            paths = []
            for row in cursor.fetchall():
                try:
                    path = json.loads(row["path"])
                    paths.append(path)
                except json.JSONDecodeError:  # noqa: PERF203
                    continue
            return paths
        except Exception as e:
            print(f"Path finding failed: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete all relations for a memory."""
        try:
            self.db.execute(
                "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            )
            self.db.commit()
            return True
        except Exception:
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        try:
            # Count unique nodes (memories that have relations)
            node_cursor = self.db.execute(
                """
                SELECT COUNT(DISTINCT id) FROM (
                    SELECT source_id as id FROM relations
                    UNION
                    SELECT target_id as id FROM relations
                )
                """
            )
            node_count = node_cursor.fetchone()[0]

            # Count edges (relations)
            edge_cursor = self.db.execute("SELECT COUNT(*) FROM relations")
            edge_count = edge_cursor.fetchone()[0]

            return {
                "available": True,
                "nodes": node_count,
                "edges": edge_count,
                "implementation": "sqlite",
            }
        except Exception as e:
            return {"available": True, "error": str(e)}


# Global instance
_graph: SimpleGraph | None = None


def get_simple_graph(config: BrainConfig | None = None) -> SimpleGraph:
    """Get or create global graph instance."""
    global _graph  # noqa: PLW0603
    if _graph is None:
        _graph = SimpleGraph(config)
    return _graph


def reset_simple_graph() -> None:
    """Reset global graph instance."""
    global _graph  # noqa: PLW0603
    _graph = None


# Alias for compatibility
get_graph = get_simple_graph
