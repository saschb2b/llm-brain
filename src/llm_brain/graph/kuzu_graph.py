"""KùzuDB graph database integration for memory relations."""

import time
from contextlib import suppress
from typing import Any, Optional

try:
    import kuzu

    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

from llm_brain.core.config import BrainConfig, get_config


class KuzuGraph:
    """Graph database manager using KùzuDB."""

    def __init__(self, config: Optional[BrainConfig] = None) -> None:
        """Initialize graph database.

        Args:
            config: Brain configuration
        """
        self.config = config or get_config()
        self._db: Optional[Any] = None
        self._conn: Optional[Any] = None

        if KUZU_AVAILABLE:
            self._init_database()

    @property
    def is_available(self) -> bool:
        """Check if KùzuDB is available."""
        return KUZU_AVAILABLE and self._db is not None

    def _init_database(self) -> None:
        """Initialize KùzuDB database and schema."""
        if not KUZU_AVAILABLE:
            return

        try:
            self._db = kuzu.Database(str(self.config.kuzu_path))
            self._conn = kuzu.Connection(self._db)
            self._create_schema()
        except Exception:
            self._db = None
            self._conn = None

    def _create_schema(self) -> None:
        """Create node and relationship tables."""
        if not self._conn:
            return

        # Create Memory node table
        with suppress(Exception):
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Memory(
                    id STRING,
                    tier STRING,
                    importance DOUBLE,
                    created_at INT64,
                    PRIMARY KEY (id)
                )
            """)

        # Create RELATES_TO relationship table
        with suppress(Exception):
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATES_TO(
                    FROM Memory TO Memory,
                    relation_type STRING,
                    weight DOUBLE,
                    created_at INT64
                )
            """)

    def add_memory_node(self, memory_id: str, tier: str, importance: float) -> bool:
        """Add a memory node to the graph.

        Args:
            memory_id: Memory ID
            tier: Memory tier
            importance: Importance score

        Returns:
            True if added successfully
        """
        if not self.is_available or self._conn is None:
            return False

        try:
            created_at = int(time.time())

            self._conn.execute(
                "CREATE (m:Memory {id: $id, tier: $tier, importance: $importance, "
                "created_at: $created_at})",
                {"id": memory_id, "tier": tier, "importance": importance, "created_at": created_at},
            )
            return True
        except Exception:
            return False

    def add_relation(
        self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0
    ) -> bool:
        """Add a relation between memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relation_type: Type of relation
            weight: Relation weight

        Returns:
            True if added successfully
        """
        if not self.is_available or self._conn is None:
            return False

        try:
            created_at = int(time.time())

            self._conn.execute(
                """
                MATCH (m1:Memory {id: $source_id}), (m2:Memory {id: $target_id})
                CREATE (m1)-[:RELATES_TO {relation_type: $type, weight: $weight, \
                    created_at: $created_at}]->(m2)
                """,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": relation_type,
                    "weight": weight,
                    "created_at": created_at,
                },
            )
            return True
        except Exception:
            return False

    def get_related_memories(
        self, memory_id: str, relation_type: Optional[str] = None, min_weight: float = 0.0
    ) -> list[dict[str, Any]]:
        """Get memories related to a given memory.

        Args:
            memory_id: Source memory ID
            relation_type: Filter by relation type
            min_weight: Minimum relation weight

        Returns:
            List of related memories with relation info
        """
        if not self.is_available or self._conn is None:
            return []

        try:
            if relation_type:
                result = self._conn.execute(
                    """
                    MATCH (m:Memory {id: $id})-[:RELATES_TO {relation_type: $type}]->(
                        related:Memory)
                    WHERE r.weight >= $min_weight
                    RETURN related.id as id, related.tier as tier,
                           related.importance as importance, r.weight as weight
                    """,
                    {"id": memory_id, "type": relation_type, "min_weight": min_weight},
                )
            else:
                result = self._conn.execute(
                    """
                    MATCH (m:Memory {id: $id})-[r:RELATES_TO]->(related:Memory)
                    WHERE r.weight >= $min_weight
                    RETURN related.id as id, related.tier as tier,
                           related.importance as importance, r.weight as weight,
                           r.relation_type as relation_type
                    """,
                    {"id": memory_id, "min_weight": min_weight},
                )

            return [dict(row) for row in result]
        except Exception:
            return []

    def multi_hop_query(
        self, start_id: str, hops: int = 2, relation_type: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Find memories reachable within N hops.

        Args:
            start_id: Starting memory ID
            hops: Number of hops
            relation_type: Filter by relation type

        Returns:
            List of reachable memories
        """
        if not self.is_available or self._conn is None:
            return []

        try:
            if relation_type:
                query = f"""
                    MATCH (m:Memory {{id: $id}})-[:RELATES_TO*1..{hops} {{
                        relation_type: $type}}]->(related:Memory)
                    RETURN DISTINCT related.id as id, related.tier as tier,
                           related.importance as importance, length(p) as hops
                """
                result = self._conn.execute(query, {"id": start_id, "type": relation_type})
            else:
                query = f"""
                    MATCH (m:Memory {{id: $id}})-[:RELATES_TO*1..{hops}]->(related:Memory)
                    RETURN DISTINCT related.id as id, related.tier as tier,
                           related.importance as importance, length(p) as hops
                """
                result = self._conn.execute(query, {"id": start_id})

            return [dict(row) for row in result]
        except Exception:
            return []

    def find_paths(self, source_id: str, target_id: str, max_length: int = 3) -> list[list[str]]:
        """Find paths between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of memory IDs)
        """
        if not self.is_available or self._conn is None:
            return []

        try:
            result = self._conn.execute(
                f"""
                MATCH p = (m1:Memory {{id: $source_id}})-[:RELATES_TO*1..{max_length}]->(
                    m2:Memory {{id: $target_id}})
                RETURN [node in nodes(p) | node.id] as path
                LIMIT 10
                """,
                {"source_id": source_id, "target_id": target_id},
            )

            return [row["path"] for row in result]
        except Exception:
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory node and its relations.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted
        """
        if not self.is_available or self._conn is None:
            return False

        try:
            self._conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})
            return True
        except Exception:
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with node and edge counts
        """
        if not self.is_available or self._conn is None:
            return {"available": False}

        try:
            node_count = 0
            edge_count = 0

            result = self._conn.execute("MATCH (m:Memory) RETURN count(m) as count")
            for row in result:
                node_count = row["count"]

            result = self._conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count")
            for row in result:
                edge_count = row["count"]

            return {"available": True, "nodes": node_count, "edges": edge_count}
        except Exception:
            return {"available": True, "error": "Could not retrieve statistics"}


# Global graph instance
_graph: Optional[KuzuGraph] = None


def get_graph(config: Optional[BrainConfig] = None) -> KuzuGraph:
    """Get or create global graph instance.

    Args:
        config: Brain configuration

    Returns:
        KuzuGraph instance
    """
    global _graph  # noqa: PLW0603
    if _graph is None:
        _graph = KuzuGraph(config)
    return _graph


def reset_graph() -> None:
    """Reset global graph instance."""
    global _graph  # noqa: PLW0603
    _graph = None
