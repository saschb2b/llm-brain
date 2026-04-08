"""FastAPI web server for brain monitoring dashboard."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llm_brain import Brain

# Paths
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="LLM-Brain Dashboard", version="0.2.0")

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


@contextmanager
def open_brain() -> Generator[Brain, None, None]:
    """Open a brain instance with automatic cleanup."""
    if _brain_path:
        brain = Brain(brain_path=_brain_path, vector_dimensions=_brain_dimensions)
    else:
        brain = Brain(vector_dimensions=_brain_dimensions)
    try:
        yield brain
    finally:
        brain.close()


def _memory_to_dict(memory: Any) -> dict[str, Any]:
    """Convert a Memory to a JSON-safe dict."""
    text = memory.raw_text or ""
    return {
        "id": memory.memory_id,
        "tier": memory.tier.value,
        "raw_text": text[:200] + "..." if len(text) > 200 else text,
        "importance": float(memory.metadata.importance_score),
        "access_count": int(memory.metadata.access_count),
        "last_accessed": memory.last_accessed.isoformat(),
        "created_at": memory.created_at.isoformat(),
    }


# --- Pages ---


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Main dashboard page."""
    return templates.TemplateResponse(request, "dashboard.html")


# --- API ---


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Get brain health status."""
    with open_brain() as brain:
        return brain.health_check()


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    """Get brain statistics."""
    with open_brain() as brain:
        return brain.stats()


@app.get("/api/metrics/live")
async def live_metrics() -> dict[str, Any]:
    """Aggregated metrics for the dashboard in a single request."""
    with open_brain() as brain:
        health_data = brain.health_check()
        stats_data = brain.stats()
        tier_counts = brain.storage.count_by_tier()
        db_size = _get_db_size(brain)

        return {
            "health": health_data,
            "stats": stats_data,
            "tier_counts": tier_counts,
            "db_size_mb": db_size,
        }


@app.get("/api/memories/recent")
async def recent_memories(limit: int = Query(default=10, ge=1, le=100)) -> list[dict[str, Any]]:
    """Get recently accessed memories (efficient single query)."""
    with open_brain() as brain:
        return [_memory_to_dict(m) for m in brain.storage.recall_recent(top_k=limit)]


@app.get("/api/memories/important")
async def important_memories(limit: int = Query(default=10, ge=1, le=100)) -> list[dict[str, Any]]:
    """Get most important memories (efficient single query)."""
    with open_brain() as brain:
        return [_memory_to_dict(m) for m in brain.storage.recall_most_important(top_k=limit)]


@app.get("/api/memories/search")
async def search_memories(
    q: str = Query(default="", min_length=0, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Search memories by text content."""
    if not q:
        with open_brain() as brain:
            return [_memory_to_dict(m) for m in brain.storage.recall_all(top_k=limit)]
    with open_brain() as brain:
        return [_memory_to_dict(m) for m in brain.storage.search_memories(q, top_k=limit)]


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete a memory by ID."""
    with open_brain() as brain:
        deleted = brain.forget(memory_id)
        return {"deleted": deleted, "memory_id": memory_id}


@app.get("/api/memories/by-tier")
async def memories_by_tier() -> dict[str, int]:
    """Get memory counts by tier."""
    with open_brain() as brain:
        return brain.storage.count_by_tier()


@app.get("/api/graph/stats")
async def graph_stats() -> dict[str, Any]:
    """Get graph database statistics."""
    with open_brain() as brain:
        return brain.graph.get_statistics()


@app.get("/api/graph/data")
async def graph_data() -> dict[str, Any]:
    """Get graph data for visualization."""
    with open_brain() as brain:
        nodes = []
        for memory in brain.storage.recall_all(top_k=200):
            text = memory.raw_text or ""
            nodes.append({
                "id": memory.memory_id,
                "label": text[:40] + "..." if len(text) > 40 else text,
                "tier": memory.tier.value,
                "importance": float(memory.metadata.importance_score),
            })

        edges = []
        try:
            graph_stats_data = brain.graph.get_statistics()
            if graph_stats_data.get("available") and graph_stats_data.get("edges", 0) > 0:
                cursor = brain.storage.db.execute("""
                    SELECT r.source_id, r.target_id, r.relation_type, r.weight
                    FROM relations r
                    WHERE r.source_id IN (SELECT memory_id FROM memories)
                      AND r.target_id IN (SELECT memory_id FROM memories)
                """)
                edges.extend(
                    {
                        "source": row["source_id"],
                        "target": row["target_id"],
                        "type": row["relation_type"],
                        "weight": float(row["weight"]),
                    }
                    for row in cursor.fetchall()
                )
        except Exception:  # noqa: S110
            pass

        return {"nodes": nodes, "edges": edges}


@app.get("/api/cognition/log")
async def cognition_log(limit: int = Query(default=20, ge=1, le=200)) -> list[dict[str, Any]]:
    """Get recent cognition log entries."""
    with open_brain() as brain:
        return brain.storage.read_cognition_log(limit=limit)


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
