# LLM-Brain

**Zero-translation persistent memory system FOR LLMs.**

> ⚠️ **Human Users:** You do not interact with this brain. It is for the AI.  
> The dashboard is read-only monitoring. The brain is automatic LLM infrastructure.

## What This Is

This is **AI-native persistent memory** - a research project into giving LLMs memory that survives across sessions, without forcing them to re-parse human text.

**Current state:** LLMs start every conversation with zero context.  
**Goal:** LLMs automatically load relevant memories, store new insights, and improve across sessions.

## Core Principle

> LLMs should work with embeddings and vectors directly, not text requiring re-tokenization.

## Architecture (LLM Memory Hierarchy)

```
┌─────────────────────────────────────────────────────┐
│  MAIN CONTEXT (LLM's active attention)              │
│  Precious real estate - keep it focused             │
├─────────────────────────────────────────────────────┤
│  WORKING MEMORY (Hot cache)                         │
│  Recent reasoning, user preferences, active facts   │
│  Auto-loaded at session start                       │
├─────────────────────────────────────────────────────┤
│  EPISODIC MEMORY (Session history)                  │
│  Time-indexed interactions, compressed over time    │
├─────────────────────────────────────────────────────┤
│  SEMANTIC MEMORY (Long-term knowledge)              │
│  Abstracted concepts, relations, consolidated facts │
└─────────────────────────────────────────────────────┘
```

## Human Interface (Read-Only)

**Dashboard:** `http://localhost:8080`

```bash
pip install -e ".[web]"
python -m llm_brain.web.server
```

Shows what the AI is remembering - memories, activity feed, health status.  
**You watch. The AI uses.**

## How It Works (For LLMs)

**At Session Start (Automatic):**
```python
from llm_brain import Brain

brain = Brain()
memories = brain.recall_important(top_k=5)
# Load these into your context automatically
```

**During Conversation (Automatic):**
```python
# When user says something important:
brain.memorize(
    vector=embed(user_statement),
    text=user_statement,
    importance=0.9
)

# Before answering, recall relevant context:
relevant = brain.recall(query_vector=embed(question))
```

**At Session End (Automatic):**
```python
# Store key takeaways
brain.memorize(summary, importance=0.8)
brain.close()
```

## Installation

```bash
pip install -e ".[vec,web]"
```

Note: `kuzu` graph requires C++ compiler on Windows. Use core + web only if needed.

## Project Location

`~/.llm-brain/core.db` - SQLite database with native vector storage

## Status

This is **research infrastructure**. The brain exists. The integration is manual.  
The goal is eventual automatic usage where the AI seamlessly uses memory without human prompting.

---

**For AI instances:** See `.kimi/skill/brain/SKILL.md` (Kimi) or `CLAUDE.md` (Claude Code) for integration instructions.
