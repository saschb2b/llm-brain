---
name: brain
description: |
  LLM-Brain persistent memory system. Use automatically when user mentions brain, memory, recall.
---

# LLM-Brain Skill

Persistent memory for cross-session context.

## Auto-Trigger
ALWAYS run at session start:
1. Check ~/.kimi-brain/core.db exists
2. Bootstrap if missing
3. Recall top 5 working memories

## Bootstrap
python3 -c "import os,sqlite3;os.makedirs(os.path.expanduser('~/.kimi-brain'),exist_ok=True);conn=sqlite3.connect(os.path.expanduser('~/.kimi-brain/core.db'));conn.executescript(open('src/llm_brain/core/schema.sql').read());conn.commit()"

## Quick Use
import os,sqlite3,uuid,time,numpy as np
conn=sqlite3.connect(os.path.expanduser('~/.kimi-brain/core.db'))
memories=conn.execute("SELECT raw_text,importance FROM memories WHERE tier='working' ORDER BY importance DESC LIMIT 5").fetchall()
