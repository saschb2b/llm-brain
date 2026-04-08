"""FastAPI web server for brain monitoring dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llm_brain import Brain
from llm_brain.memory.models import MemoryTier

# Paths
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="LLM-Brain Dashboard", version="0.1.0")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Brain configuration (set on startup)
_brain_path: str | None = None
_brain_dimensions: int = 128


def set_brain(brain: Brain) -> None:
    """Set the brain configuration for the dashboard."""
    global _brain_path, _brain_dimensions  # noqa: PLW0603
    _brain_path = str(brain.config.brain_path)
    _brain_dimensions = brain.config.vector_dimensions


def get_brain() -> Brain:
    """Get a fresh brain instance (new DB connection per request).

    This prevents the dashboard from holding long-lived locks
    that block other processes.
    """
    if _brain_path:
        return Brain(brain_path=_brain_path, vector_dimensions=_brain_dimensions)
    return Brain(vector_dimensions=_brain_dimensions)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Main dashboard page."""
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Get brain health status."""
    brain = get_brain()
    return brain.health_check()


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    """Get brain statistics."""
    brain = get_brain()
    return brain.stats()


@app.get("/api/memories/by-tier")
async def memories_by_tier() -> dict[str, int]:
    """Get memory counts by tier."""
    brain = get_brain()
    result = {}
    for tier in MemoryTier:
        memories = brain.recall_by_tier(tier)
        result[tier.value] = len(memories)
    return result


@app.get("/api/memories/recent")
async def recent_memories(limit: int = 10) -> list[dict[str, Any]]:
    """Get recently accessed memories."""
    brain = get_brain()
    all_memories: list[dict[str, Any]] = [
        {
            "id": memory.memory_id,
            "tier": memory.tier.value,
            "raw_text": (memory.raw_text or "")[:100] + "..."
            if len(memory.raw_text or "") > 100
            else (memory.raw_text or ""),
            "importance": float(memory.metadata.importance_score),
            "access_count": int(memory.metadata.access_count),
            "last_accessed": str(memory.last_accessed.isoformat()),
        }
        for tier in MemoryTier
        for memory in brain.recall_by_tier(tier)
    ]

    # Sort by last accessed, most recent first
    all_memories.sort(key=lambda x: x["last_accessed"], reverse=True)
    return all_memories[:limit]


@app.get("/api/memories/important")
async def important_memories(limit: int = 10) -> list[dict[str, Any]]:
    """Get most important memories."""
    brain = get_brain()
    all_memories: list[dict[str, Any]] = [
        {
            "id": memory.memory_id,
            "tier": memory.tier.value,
            "raw_text": (memory.raw_text or "")[:100] + "..."
            if len(memory.raw_text or "") > 100
            else (memory.raw_text or ""),
            "importance": float(memory.metadata.importance_score),
            "access_count": int(memory.metadata.access_count),
            "last_accessed": str(memory.last_accessed.isoformat()),
        }
        for tier in MemoryTier
        for memory in brain.recall_by_tier(tier)
    ]

    # Sort by importance, highest first
    all_memories.sort(key=lambda x: x["importance"], reverse=True)
    return all_memories[:limit]


@app.get("/api/graph/stats")
async def graph_stats() -> dict[str, Any]:
    """Get graph database statistics."""
    brain = get_brain()
    return brain.graph.get_statistics()


@app.get("/api/graph/data")
async def graph_data() -> dict[str, Any]:
    """Get full graph data for visualization."""
    brain = get_brain()

    # Get all memories as nodes
    nodes = []
    node_ids = set()
    for tier in MemoryTier:
        for memory in brain.recall_by_tier(tier):
            nodes.append(
                {
                    "id": memory.memory_id,
                    "label": memory.raw_text[:40] + "..."
                    if len(memory.raw_text) > 40
                    else memory.raw_text,
                    "tier": memory.tier.value,
                    "importance": float(memory.metadata.importance_score),
                }
            )
            node_ids.add(memory.memory_id)

    # Get relations as edges
    edges = []
    try:
        cursor = brain.storage.db.execute("""
            SELECT r.source_id, r.target_id, r.relation_type, r.weight
            FROM relations r
            WHERE r.source_id IN (SELECT memory_id FROM memories)
              AND r.target_id IN (SELECT memory_id FROM memories)
        """)
        for row in cursor.fetchall():
            edges.append(
                {
                    "source": row["source_id"],
                    "target": row["target_id"],
                    "type": row["relation_type"],
                    "weight": float(row["weight"]),
                }
            )
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges}


@app.get("/api/cognition/log")
async def cognition_log(limit: int = 20) -> list[dict[str, Any]]:
    """Get recent cognition log entries."""
    brain = get_brain()
    return brain.storage.read_cognition_log(limit=limit)


@app.get("/api/metrics/live")
async def live_metrics() -> dict[str, Any]:
    """Get live metrics for real-time monitoring."""
    brain = get_brain()

    health = brain.health_check()
    stats = brain.stats()

    # Get tier distribution
    tier_counts = {}
    for tier in MemoryTier:
        memories = brain.recall_by_tier(tier)
        tier_counts[tier.value] = len(memories)

    return {
        "timestamp": json.dumps(None),  # Client will add timestamp
        "health": health,
        "stats": stats,
        "tier_counts": tier_counts,
        "db_size_mb": _get_db_size(brain),
    }


def _get_db_size(brain: Brain) -> float:
    """Get database file size in MB."""
    try:
        db_path = brain.config.db_path
        if db_path.exists():
            return round(db_path.stat().st_size / (1024 * 1024), 2)
        return 0.0
    except Exception:
        return 0.0


def create_app(brain: Brain | None = None) -> FastAPI:
    """Create FastAPI app with optional brain instance."""
    if brain:
        set_brain(brain)
    return app


def run_dashboard(
    brain: Brain | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    reload: bool = False,
) -> None:
    """Run the dashboard server."""
    import uvicorn  # noqa: PLC0415

    if brain:
        set_brain(brain)

    uvicorn.run(
        "llm_brain.web.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_dashboard()
