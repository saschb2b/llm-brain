-- LLM-Brain Database Schema v1.0

-- Core memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL CHECK(tier IN ('working', 'episodic', 'semantic')),
    created_at INTEGER NOT NULL,  -- Unix timestamp
    accessed_at INTEGER NOT NULL, -- Unix timestamp
    vector BLOB,                  -- Float16 array (raw bytes)
    content_hash TEXT UNIQUE,     -- For deduplication
    importance REAL DEFAULT 0.5,  -- 0.0-1.0, LLM-assigned
    confidence REAL DEFAULT 0.5,  -- 0.0-1.0, certainty of fact
    compression_ratio REAL DEFAULT 1.0,  -- 1.0 = no compression
    access_count INTEGER DEFAULT 0,      -- For LRU eviction
    decay_rate REAL DEFAULT 0.01,        -- Time-based forgetting
    metadata JSON,                -- Flexible metadata storage
    raw_text TEXT,                -- Optional human-readable for debugging
    token_map JSON,               -- Source token positions
    context_window_id TEXT        -- Session identifier
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);

-- Vector search virtual table (created via sqlite-vec in Python)
-- Note: This is created programmatically due to dynamic dimensions

-- Knowledge graph edges (stored in SQLite for quick lookup, KùzuDB for traversal)
CREATE TABLE IF NOT EXISTS relations (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (source_id, target_id, relation_type),
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

-- Audit/observability log
CREATE TABLE IF NOT EXISTS cognition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    operation TEXT NOT NULL CHECK(operation IN (
        'retrieve', 'store', 'compress', 'contradict',
        'evict', 'promote', 'demote', 'forget'
    )),
    memory_id TEXT,
    context_query TEXT,           -- What was being searched for
    latency_ms INTEGER,
    metadata JSON,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_cognition_log_timestamp ON cognition_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_cognition_log_operation ON cognition_log(operation);
CREATE INDEX IF NOT EXISTS idx_cognition_log_memory ON cognition_log(memory_id);

-- Session management
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    context_window_size INTEGER,
    hot_memory_ids TEXT,          -- JSON array of UUIDs loaded at start
    metadata JSON
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

-- Configuration table for schema versioning
CREATE TABLE IF NOT EXISTS brain_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Insert schema version
INSERT OR REPLACE INTO brain_meta (key, value, updated_at)
VALUES ('schema_version', '1.0', CAST(strftime('%s', 'now') AS INTEGER));
