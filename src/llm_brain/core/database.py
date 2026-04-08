"""Database initialization and management for LLM-Brain."""

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np

from llm_brain.core.config import BrainConfig, get_config


class Database:
    """SQLite database manager with sqlite-vec extension."""

    def __init__(self, config: Optional[BrainConfig] = None) -> None:
        """Initialize database connection.

        Args:
            config: Brain configuration. Uses global config if not provided.
        """
        self.config = config or get_config()
        self._conn: Optional[sqlite3.Connection] = None
        self._vec_loaded = False

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        """Create and configure SQLite connection."""
        # Ensure directory exists
        self.config.brain_path.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.config.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Load sqlite-vec extension if available
        self._load_vec_extension(conn)

        return conn

    def _load_vec_extension(self, conn: sqlite3.Connection) -> bool:
        """Attempt to load sqlite-vec extension.

        Returns:
            True if extension loaded successfully
        """
        if self._vec_loaded:
            return True

        try:
            # Try to import sqlite_vec Python package
            spec = importlib.util.find_spec("sqlite_vec")
            if spec is not None:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._vec_loaded = True
                return True
        except Exception:
            pass

        # Fallback: try loading from common paths
        vec_paths = [
            "vec0",
            "sqlite_vec",
            "/usr/lib/sqlite3/vec0.so",
            "/usr/local/lib/sqlite3/vec0.so",
        ]

        for path in vec_paths:
            try:
                conn.enable_load_extension(True)
                conn.load_extension(path)
                conn.enable_load_extension(False)
                self._vec_loaded = True
                return True
            except sqlite3.OperationalError:
                continue

        return False

    def initialize_schema(self) -> None:
        """Initialize database schema from SQL file."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema = f.read()

        self.connection.executescript(schema)
        self.connection.commit()

        # Create vector virtual table if sqlite-vec is available
        if self._vec_loaded:
            self._create_vector_table()

    def _create_vector_table(self) -> None:
        """Create virtual table for vector search."""
        # Check if table exists
        cursor = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
        )
        if cursor.fetchone():
            return

        dims = self.config.vector_dimensions
        # Create virtual table using sqlite-vec syntax
        try:
            self.connection.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dims}]
                )
            """)
            self.connection.commit()
        except sqlite3.OperationalError as e:
            # sqlite-vec might not be available or have different syntax
            print(f"Warning: Could not create vector table: {e}")

    def execute(self, query: str, parameters: Optional[tuple[Any, ...]] = None) -> sqlite3.Cursor:
        """Execute SQL query.

        Args:
            query: SQL query string
            parameters: Query parameters

        Returns:
            SQLite cursor
        """
        if parameters:
            return self.connection.execute(query, parameters)
        return self.connection.execute(query)

    def executemany(self, query: str, parameters: list[tuple[Any, ...]]) -> sqlite3.Cursor:
        """Execute SQL query with multiple parameter sets.

        Args:
            query: SQL query string
            parameters: List of parameter tuples

        Returns:
            SQLite cursor
        """
        return self.connection.executemany(query, parameters)

    def commit(self) -> None:
        """Commit current transaction."""
        self.connection.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def is_initialized(self) -> bool:
        """Check if database has been initialized."""
        try:
            cursor = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            )
            return cursor.fetchone() is not None
        except sqlite3.Error:
            return False

    def get_schema_version(self) -> Optional[str]:
        """Get current schema version."""
        try:
            cursor = self.connection.execute(
                "SELECT value FROM brain_meta WHERE key = 'schema_version'"
            )
            row = cursor.fetchone()
            return row["value"] if row else None
        except sqlite3.Error:
            return None


# Global database instance
_db: Optional[Database] = None


def get_database(config: Optional[BrainConfig] = None) -> Database:
    """Get or create global database instance.

    Args:
        config: Brain configuration

    Returns:
        Database instance
    """
    global _db
    if _db is None:
        _db = Database(config)
    return _db


def reset_database() -> None:
    """Reset global database instance (useful for testing)."""
    global _db
    if _db:
        _db.close()
    _db = None


def vector_to_blob(vector: np.ndarray) -> bytes:
    """Convert numpy array to float16 blob.

    Args:
        vector: Numpy array of floats

    Returns:
        Bytes blob in float16 format
    """
    return vector.astype(np.float16).tobytes()


def blob_to_vector(blob: bytes, dimensions: int) -> np.ndarray:
    """Convert float16 blob to numpy array.

    Args:
        blob: Bytes blob
        dimensions: Expected vector dimensions

    Returns:
        Numpy array of float32 (for computation)
    """
    return np.frombuffer(blob, dtype=np.float16, count=dimensions).astype(np.float32)
