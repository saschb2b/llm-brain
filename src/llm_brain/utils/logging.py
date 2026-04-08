"""Cognition logging for observability."""

import json
from datetime import datetime, timezone
from typing import Any, Optional

from llm_brain.core.config import BrainConfig, get_config


class CognitionLogger:
    """Logger for cognitive operations audit trail."""

    def __init__(self, config: Optional[BrainConfig] = None) -> None:
        """Initialize logger.

        Args:
            config: Brain configuration
        """
        self.config = config or get_config()
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Create log file if it doesn't exist."""
        log_path = self.config.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not log_path.exists():
            log_path.touch()

    def log(
        self,
        operation: str,
        memory_id: Optional[str] = None,
        context: Optional[str] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write entry to cognition log.

        Args:
            operation: Operation type
            memory_id: Related memory ID
            context: Query context
            latency_ms: Operation latency
            metadata: Additional metadata
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "memory_id": memory_id,
            "context": context,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
        }

        with open(self.config.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_logs(
        self, n: Optional[int] = None, operation: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Read log entries.

        Args:
            n: Number of entries to read (most recent)
            operation: Filter by operation type

        Returns:
            List of log entries
        """
        if not self.config.log_path.exists():
            return []

        entries: list[dict[str, Any]] = []

        with open(self.config.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if operation is None or entry.get("operation") == operation:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue

        if n:
            entries = entries[-n:]

        return entries

    def get_stats(self) -> dict[str, Any]:
        """Get log statistics.

        Returns:
            Dictionary with statistics
        """
        entries = self.read_logs()

        if not entries:
            return {"total_operations": 0}

        operations: dict[str, int] = {}
        latencies: list[int] = []

        for entry in entries:
            op = entry.get("operation", "unknown")
            operations[op] = operations.get(op, 0) + 1

            if entry.get("latency_ms"):
                latencies.append(entry["latency_ms"])

        stats = {
            "total_operations": len(entries),
            "operations_breakdown": operations,
        }

        if latencies:
            stats["avg_latency_ms"] = sum(latencies) / len(latencies)
            stats["max_latency_ms"] = max(latencies)
            stats["min_latency_ms"] = min(latencies)

        return stats
