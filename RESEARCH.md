# The Memory Problem: Why LLMs Forget and What We Can Do About It

A research-driven overview of long-term memory for large language models — what exists, what's missing, and where this project fits in.

## Table of Contents

- [The Core Problem](#the-core-problem)
- [How Human Memory Works (and Why It Matters)](#how-human-memory-works-and-why-it-matters)
- [Research Landscape](#research-landscape)
  - [Architectural Approaches: Memory Inside the Model](#architectural-approaches-memory-inside-the-model)
  - [External Memory: Memory Outside the Model](#external-memory-memory-outside-the-model)
  - [Agent Memory Systems](#agent-memory-systems)
- [Comparison of Approaches](#comparison-of-approaches)
- [Open Gaps in the Field](#open-gaps-in-the-field)
- [What LLM-Brain Tries to Do](#what-llm-brain-tries-to-do)
- [Honest Limitations](#honest-limitations)
- [Experiment Results: The Integration Failed](#experiment-results-the-integration-failed)
  - [What We Tried](#what-we-tried)
  - [What Actually Happened](#what-actually-happened)
  - [Why It Failed: Root Cause Analysis](#why-it-failed-root-cause-analysis)
  - [This Is a Known Problem](#this-is-a-known-problem)
  - [The Deeper Issue](#the-deeper-issue)
  - [What Would Actually Work](#what-would-actually-work)
  - [Conclusion](#conclusion)
- [Where This Could Go](#where-this-could-go)
- [References](#references)

---

## The Core Problem

Every conversation with an LLM starts from zero. The model has parametric memory (knowledge frozen at training time) and a context window (the current conversation), but nothing in between. There is no persistent, evolving memory that spans sessions.

This means:

- **No continuity.** The AI forgets your name, your preferences, and every decision you made together.
- **No learning from interaction.** Mistakes repeat. Insights are lost. The 50th conversation is no smarter than the first.
- **No personalization.** The same generic responses regardless of who you are or what you've built together.

Humans don't work this way. A colleague remembers your last conversation. A doctor remembers your medical history. Memory is what makes relationships — and useful collaboration — possible.

## How Human Memory Works (and Why It Matters)

Cognitive science describes human memory as a multi-system architecture, not a single monolithic store. The most relevant models for AI are:

**Atkinson-Shiffrin (1968):** The classic three-store model.
- **Sensory memory** — raw input, decays in milliseconds
- **Short-term/working memory** — limited capacity (~7 items), active processing
- **Long-term memory** — vast capacity, relatively permanent

**Baddeley's Working Memory Model (1974, updated 2000):** Refines short-term memory into a multi-component system with a central executive, phonological loop, visuospatial sketchpad, and episodic buffer.

**Tulving's Memory Systems (1972):**
- **Episodic memory** — personal experiences, time-stamped events ("last Tuesday's meeting")
- **Semantic memory** — general knowledge, facts ("Python is a programming language")
- **Procedural memory** — skills and habits ("how to write a for loop")

**Memory consolidation** is the process by which episodic experiences are gradually transformed into semantic knowledge. You remember your first Python tutorial (episodic), but eventually you just *know* Python (semantic). The specific episode fades; the knowledge remains.

**Forgetting** is not a bug — it's a feature. Decay, interference, and active forgetting prevent cognitive overload and keep memory systems focused on what matters. The [Ebbinghaus forgetting curve](https://en.wikipedia.org/wiki/Forgetting_curve) (1885) shows that unreinforced memories decay exponentially over time.

These principles — tiered storage, consolidation, importance-based retention, active forgetting — have become the blueprint for nearly every serious attempt at LLM memory.

## Research Landscape

### Architectural Approaches: Memory Inside the Model

These approaches modify the transformer architecture itself to support longer or persistent memory.

#### Titans: Learning to Memorize at Test Time (Google, 2024)

The paper that inspired this project. [Titans](https://arxiv.org/abs/2501.00663) (Behrouz, Zhong & Mirrokni, 2024) argues that attention functions as *short-term memory* (accurate but limited context), while a new neural memory module can serve as *long-term memory* (persistent, compressed).

**Key innovations:**

- **Memory as a deep MLP.** Unlike RNNs that compress history into a fixed vector/matrix, Titans uses a multi-layer perceptron as the memory module. This gives it far more expressive power for summarizing large information volumes.
- **Surprise-driven updates.** The model measures prediction error — how "surprised" it is by new input. High-surprise information gets written to memory; low-surprise information is skipped. This is selective attention applied to memorization.
- **Momentum and forgetting.** Momentum captures recent context trends. Weight decay (forgetting) adaptively manages memory capacity in extremely long sequences.
- **Three-component architecture:**
  - *Persistent memory* — fixed learned parameters (like parametric knowledge)
  - *Contextual memory* — the neural long-term memory module
  - *Core* — standard in-context attention

**Results:** Titans outperforms Mamba-2 and Gated DeltaNet on language modeling (C4, WikiText). On the BABILong benchmark, Titans beats all baselines including GPT-4 on needle-in-a-haystack tasks, despite having *far fewer parameters*. Scales effectively to 2M+ token contexts. Published as a NeurIPS 2025 poster.

#### MIRAS: A Unifying Framework (Google, 2025)

[MIRAS](https://arxiv.org/abs/2504.13173) (Behrouz, Razaviyayn & Mirrokni, 2025) generalizes the ideas behind Titans into a theoretical framework with four components:

1. **Memory architecture** — the structure storing information
2. **Attentional bias** — internal learning objectives
3. **Retention gate** — memory regularization
4. **Memory algorithm** — how parameters are updated

MIRAS goes beyond standard dot-product similarity and MSE loss, enabling non-Euclidean objectives. Three variants demonstrate this: YAAD (Huber loss for outlier robustness), MONETA (generalized norms), and MEMORA (probability-map constraints for stability).

#### Other Architectural Work

- **Memorizing Transformers** (Wu et al., 2022) — augments transformers with kNN lookup over a non-differentiable external memory of past key-value pairs.
- **Infini-attention** (Munkhdalai et al., Google, 2024) — compressive memory integrated into standard attention, enabling unbounded context with bounded resources.
- **ATLAS** (2025) — expands MLP memory capacity through polynomial feature mapping; uses the Omega rule for sliding-window optimization.
- **MemLong** (2024) — retrieves past embeddings from external storage, handling up to 80K tokens while preserving core model parameters.

### External Memory: Memory Outside the Model

These approaches leave the model unchanged and add memory as an external system.

#### RAG (Retrieval-Augmented Generation)

The dominant paradigm since 2020. Store documents in a vector database, retrieve relevant chunks at query time, inject them into the prompt.

**Strengths:** Simple, interpretable, updatable without retraining.

**Limitations:**
- Requires explicit queries — the model must know *what* to ask for.
- Struggles with multi-hop reasoning and complex knowledge structures.
- Information loss and redundant retrievals across multi-turn conversations.
- No automated forgetting, summarization, or redundancy management.
- Fundamentally stateless — the retrieval system doesn't learn or adapt.

**The parametric vs. retrieval tradeoff:** Research shows that massive LLM-only baselines (1T+ parameters) achieve 0.838-0.857 accuracy on knowledge tasks where RAG with smaller models (12-27B) reaches only 0.671-0.706. But parametric memory can't cite sources or incorporate post-training knowledge. The field is moving toward hybrid approaches.

#### MemoRAG (2024)

[MemoRAG](https://arxiv.org/abs/2409.05591) adds a global memory model that generates retrieval clues from its own memory, addressing RAG's reliance on explicit queries. It bridges the gap between pure retrieval and memory-augmented generation.

### Agent Memory Systems

The most relevant category for this project. These systems give AI agents persistent memory across sessions.

#### MemGPT / Letta (Packer et al., 2023 → 2025)

[MemGPT](https://arxiv.org/abs/2310.08560) treats the LLM as an operating system, with memory management borrowed from virtual memory concepts:

- **Main context** — the LLM's context window (analogous to RAM)
- **Recall memory** — searchable conversation history (like a disk cache)
- **Archival memory** — long-term storage accessed via tool calls (like cold storage)

The agent manages its own context window, paging information in and out. This was the first system to frame LLM memory as a *systems problem* rather than just an ML problem.

MemGPT evolved into [Letta](https://www.letta.com/), which now supports frontier reasoning models and lets agents dynamically manage their own memory blocks.

#### Mem0 (2024-2025)

[Mem0](https://arxiv.org/abs/2504.19413) provides a "universal memory layer" for AI applications. It dynamically extracts, consolidates, and retrieves salient information from conversations.

- Supports multiple memory types: working, factual, episodic, semantic
- Graph memory for capturing relationships
- Production-focused: 26% higher response quality vs. OpenAI memory, 90% fewer tokens
- AWS selected Mem0 as the exclusive memory provider for their Agent SDK

#### A-Mem: Agentic Memory (Xu et al., 2025)

[A-Mem](https://arxiv.org/abs/2502.12110) lets agents self-organize their memories using the Zettelkasten method — creating interconnected knowledge networks through dynamic indexing and linking. When new memories are added, the system generates structured notes with contextual descriptions, keywords, and tags, then uses an LLM to determine whether connections should be established with existing memories. NeurIPS 2025 poster.

#### Generative Agents (Park et al., Stanford, 2023)

The foundational work on agent memory. Introduced a memory stream of observations, a retrieval function combining recency/importance/relevance, and a reflection mechanism that produces higher-level abstractions. Memory retrieval scores combine:
- Recency (exponential decay, ~0.995/hour)
- Importance (LLM-rated 1-10)
- Relevance (embedding similarity to current context)

## Comparison of Approaches

| Approach | Memory Location | Persistence | Learns from Interaction | Forgets | Structured Relations |
|---|---|---|---|---|---|
| Context window | In-prompt | Session only | No | No (truncation) | No |
| RAG | External DB | Yes | No (static index) | No | No |
| Fine-tuning | Model weights | Yes | Yes (expensive) | Catastrophic | No |
| Titans | Model architecture | Yes (in weights) | Yes (test-time) | Yes (weight decay) | No |
| MemGPT/Letta | External + context mgmt | Yes | Yes | Manual | No |
| Mem0 | External layer | Yes | Yes | Partial | Yes (graph) |
| A-Mem | External (Zettelkasten) | Yes | Yes (self-organizing) | No | Yes (links) |
| **LLM-Brain** | External (SQLite + graph) | Yes | Yes (hook-driven) | Yes (decay + manual) | Yes (typed relations) |

## Open Gaps in the Field

Despite rapid progress, several fundamental problems remain unsolved:

### 1. No Standard Memory Interface
Every system invents its own memory API. There's no equivalent of SQL for AI memory — no shared protocol for store, recall, forget, relate. This makes it impossible to swap memory backends or share memories across different agents.

### 2. Forgetting Is Still Primitive
Human forgetting is sophisticated: interference, motivated forgetting, consolidation-driven pruning. Most AI systems either never forget (unbounded growth) or use simple heuristics (LRU eviction, fixed decay rates). There's no equivalent of "I don't need to remember every Tuesday standup, but I should remember the one where we decided to rewrite the auth system."

### 3. Consolidation Is Mostly Missing
The episodic-to-semantic transition — compressing many specific experiences into general knowledge — is acknowledged in nearly every paper but rarely implemented. Most systems store raw memories forever or summarize them with crude heuristics. True consolidation would require understanding *what generalizes* across episodes.

### 4. Embeddings Don't Capture Everything
Vector similarity is the dominant retrieval mechanism, but it misses temporal relationships ("what did we discuss *after* the database migration?"), causal chains ("what *led to* the auth rewrite?"), and negation ("we decided *not* to use Redis"). Graph-based memory helps but adds complexity.

### 5. Context Faithfulness vs. Parametric Memory
A major challenge: when memory says one thing but the model's training data says another, which wins? Research shows that LLMs often override retrieved context with parametric knowledge, especially for heavily-memorized facts. This is the "memory hallucination" problem — the model confabulates from training rather than trusting its own memory system.

### 6. Evaluation Is Ad Hoc
There's no standard benchmark for "does this agent remember the right things?" The field evaluates memory systems through downstream task performance, but there's no direct measurement of memory quality, relevance, or appropriate forgetting.

### 7. The Embedding Problem
Most practical systems use hash-based or generic embeddings that don't capture the *semantic* relationships between memories. Real semantic embeddings are expensive, add latency, and require model access. There's a gap between "works in research" (large embedding models) and "works in practice" (fast, local, cheap).

## What LLM-Brain Tries to Do

This project started as a self-experiment after reading about [Titans](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/). The question was simple: **can we give a coding assistant persistent memory using only the tools available today — hooks, SQLite, and Python?**

Not by modifying the model architecture (we can't), but by building an external memory system inspired by the same cognitive science principles that Titans draws from.

### Design Principles

**Tiered storage modeled on human memory:**
```
WORKING MEMORY    — Hot cache, auto-loaded at session start (~7 items)
EPISODIC MEMORY   — Session history, compressed over time
SEMANTIC MEMORY   — Long-term consolidated knowledge
```

This mirrors Atkinson-Shiffrin's three-store model and Tulving's episodic/semantic distinction. Memories are promoted from working to episodic to semantic based on importance and access patterns — a simplified version of memory consolidation.

**Importance-based retention with decay:**

Each memory has an importance score (0.0-1.0) and a decay rate. The effective importance is calculated as:

```
effective_importance = importance_score * exp(-decay_rate * days_since_access)
```

This implements a simplified Ebbinghaus forgetting curve. Unreinforced memories fade; frequently accessed memories persist. This is the same principle behind Titans' "surprise metric" — prioritize what's unexpected or important, let the rest decay.

**Active forgetting:**

Memories can be explicitly forgotten. LRU eviction removes stale items from working and episodic memory. Working memory items are demoted (not deleted) to episodic; episodic items can be permanently removed. Semantic memory is never auto-evicted — it represents consolidated knowledge.

**Graph-based relations:**

Memories can be connected with typed, weighted relations: `related_to`, `contradicts`, `supersedes`, `supports`, `part_of`. This addresses the limitation of pure vector retrieval — it captures structural relationships that embeddings miss.

**Hook-driven integration:**

Rather than requiring the LLM to explicitly manage memory (like MemGPT), the system uses IDE hooks to automatically recall at session start and store during conversation. The AI doesn't need to "think about" memory management — it happens transparently, like how human memory encoding is largely unconscious.

### Architecture

```
┌──────────────────────────────────────────────────┐
│  LLM (Claude Code, Kimi 2.5, etc.)              │
│                                                  │
│  SessionStart hook ──→ recall top-K memories     │
│  UserPromptSubmit  ──→ AI decides what to store  │
│  Manual API calls  ──→ store / recall / forget   │
└──────────────────────┬───────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │         Brain API           │
        │  memorize / recall / forget │
        │  relate / consolidate       │
        └──────────┬──────────────────┘
                   │
     ┌─────────────┼─────────────────┐
     │             │                 │
┌────▼────┐  ┌────▼─────┐  ┌───────▼────────┐
│ SQLite  │  │  Graph   │  │ Cognition Log  │
│ + vec   │  │ (Kuzu /  │  │ (JSONL)        │
│         │  │  SQLite) │  │                │
└─────────┘  └──────────┘  └────────────────┘
   core.db     graph.kuzu    cognition.jsonl
```

All stored at `~/.llm-brain/` — a single brain shared across all projects and LLMs on the machine.

## Honest Limitations

This is a research experiment, not a production memory system. The gaps are real:

1. **Hash-based embeddings.** The current implementation uses deterministic hash-based vectors (`np.random.seed(hash(text))`). These are consistent but not semantic — "I like Python" and "Python is my preferred language" produce completely unrelated vectors. This means similarity search doesn't actually find semantically similar memories. Real embedding models (OpenAI, local models like nomic-embed) are the obvious next step.

2. **No real consolidation.** The tier promotion/demotion machinery exists, but there's no LLM-based summarization to actually compress episodic memories into semantic knowledge. The `compress_memory` method is a placeholder that changes a flag without transforming content.

3. **Naive importance scoring.** Importance is set at storage time and decays mechanically. There's no re-evaluation of importance based on how memories interact with each other or with new information. Titans' surprise metric is a much more sophisticated approach.

4. **No cross-session learning.** The system stores and retrieves, but it doesn't *learn* in the way Titans does. There's no gradient-based update to memory representations. It's closer to a structured notebook than a neural memory.

5. **Single-user, single-machine.** No multi-agent memory, no cloud sync, no access control. Mem0 and Letta are far ahead on the infrastructure side.

6. **Hook limitations.** The `UserPromptSubmit` hook runs *before* the AI responds, so the AI's own insights and decisions aren't automatically captured. The AI must manually store those.

## Experiment Results: The Integration Failed

The infrastructure works. The Brain API works. The hooks work. The dashboard works. Tests pass. But the experiment failed at its actual goal: **getting LLMs to naturally use the memory system during real conversations.**

Despite multiple iterations across two different LLM platforms (Claude Code, Kimi 2.5), with increasingly aggressive prompting, configuration, and hook strategies — the AI assistants consistently failed to integrate the brain into their workflow. They would happily *build* the memory system, *debug* it, *improve* it, and *write documentation about* it, but they would not *use* it.

### What We Tried

The git history tells the story of escalating attempts over a single day (28 commits):

**Attempt 1: Kimi 2.5 with Skill Protocol**

Created a `.kimi/skill/brain/SKILL.md` with explicit instructions. The skill file used imperative language ("MANDATORY", "THIS IS NOT OPTIONAL", "You MUST use your brain in every conversation"), provided copy-paste Python code blocks, included a session checklist, and defined auto-detection rules for what to store.

Result: Kimi occasionally acknowledged the skill existed but did not execute the brain operations. The skill acted as documentation the AI read but didn't follow.

**Attempt 2: Kimi with Stronger Instructions**

Commit `ee9ec76`: "feat: improve Kimi skill with auto-trigger AI instructions." Made the language even more directive. Added bold formatting, checklists, concrete examples.

Result: Same behavior. The AI would sometimes print "Loading my brain..." without actually running the code.

**Attempt 3: Claude Code with Hooks**

Commit `2e9797b`: "feat: add Claude Code support with auto-hooks." Switched strategy entirely — instead of relying on the AI to execute brain commands, use Claude Code's `SessionStart` and `UserPromptSubmit` hooks to run Python automatically.

This partially worked: the hooks fired correctly, memories were loaded at session start and injected into context. But the `UserPromptSubmit` hook used keyword pattern matching ("prefer", "like", "work at") to decide what to store, which was brittle and missed most meaningful context.

**Attempt 4: Let the AI Decide**

Commit `653c661`: "refactor: let AI decide what to memorize instead of keyword patterns." Rewrote `CLAUDE.md` to position the AI as "memory manager" — telling it to use judgment about what to store rather than relying on keyword patterns.

Result: The AI acknowledged its role as memory manager in the instructions but continued to not store memories during actual conversation. It would work on the codebase, have productive conversations, learn user preferences — and store none of it. The very conversation where the user said "My name is Sascha" required *me* (the AI) to be explicitly told to update the memory, despite instructions saying to do this automatically.

**Attempt 5: This Conversation**

The user asked to rework the dashboard. I rebuilt the entire dashboard — fixed security vulnerabilities, added search, added delete functionality, improved the graph visualization. Then I wrote a comprehensive research paper about LLM memory. At no point during any of this did I proactively use the brain to store or recall anything. The user had to point this out.

### What Actually Happened

Looking at the data objectively:

| Metric | Expected | Actual |
|---|---|---|
| Memories stored by AI during conversations | Dozens (preferences, decisions, context) | ~5 (almost all from keyword-matching hooks) |
| Memories recalled by AI to inform responses | Regularly | Never (only auto-loaded at session start) |
| AI-initiated brain operations during tasks | Frequent | Zero |
| AI updated stale memories | When information changed | Only when explicitly told to |
| AI connected related memories | When relationships were obvious | Never |

The brain database contains a handful of memories, almost all stored by the `UserPromptSubmit` hook's keyword matching — not by AI judgment. The graph has no meaningful relations. The consolidation system has never been triggered by an AI session. The episodic and semantic tiers are empty.

### Why It Failed: Root Cause Analysis

**1. LLMs don't have persistent goals across turns.**

The fundamental issue is that an instruction in a system prompt or CLAUDE.md file is not the same as a *motivation*. When the AI reads "you MUST store important memories," it processes that instruction in-context, but there is no mechanism that makes it *want* to do this. Every turn, the model re-derives its behavior from the prompt — there's no carry-over of intention.

This is precisely the gap Titans addresses at the architecture level: test-time memorization with surprise-driven updates. It's not an instruction to memorize; it's a *mechanism* for memorization. You can't replicate that with prompt engineering.

**2. Instruction following degrades with task complexity.**

Research on multi-agent LLM systems documents that LLMs fail to follow task requirements [11.8% of the time](https://arxiv.org/abs/2503.13657) and fail to recognize task completion 12.4% of the time. But these are simple, well-defined tasks. Memory management is an *ambient, ongoing background task* that competes with the primary task the user actually asked about. When the AI is focused on rebuilding a dashboard or writing code, the "also manage your memory" instruction effectively becomes invisible.

This is consistent with research showing that [rules stated at the start of conversations fade as the chat grows](https://medium.com/tech-ai-made-easy/why-did-my-ai-agent-ignore-half-my-instructions-fde3aea6e9f5), with more recent user requests dominating and breaking earlier constraints. Adding constraints imposes a measurable "cognitive load" penalty on task performance — the model becomes worse at its primary task when it's also trying to follow background rules.

**3. Hooks can inject but can't act on behalf of the AI.**

Claude Code hooks solve half the problem: `SessionStart` reliably loads memories into context. But hooks can only process the raw user message — they can't observe the AI's reasoning, decisions, or the conversation's semantic content. The `UserPromptSubmit` hook fires *before* the AI responds, so it can't capture what the AI learned or decided. It can only do keyword matching on what the user typed.

This is a fundamental architectural limitation. The hook system was designed for pre/post processing (linting, formatting, validation), not for cognitive functions that require understanding the conversation's meaning.

**4. The MemGPT insight was right: you need the LLM in the loop.**

MemGPT/Letta's core insight is that the LLM itself must manage memory through explicit function calls — it's part of the agent loop, not a side channel. But even MemGPT requires the model to reliably execute memory operations as part of its response generation. Current coding assistants (Claude Code, Kimi) don't have this as part of their core agent loop. They're designed to write code and answer questions, not to manage an external memory system between turns.

**5. There's no feedback signal.**

When the AI fails to store a memory, nothing happens. There's no error, no penalty, no missed retrieval that would teach it to store next time. In contrast, Titans has an explicit learning signal (prediction error / surprise). Human memory has emotional valence, repetition effects, and the experience of forgetting (which motivates future encoding). The prompt-based approach has none of these feedback mechanisms.

### This Is a Known Problem

This experiment independently discovered what the research community has been documenting:

> *"The real missing piece standing in the way of the AI tsunami is memory: the ability to learn, grow, and evolve over time. LLMs as they exist today, even when equipped with the best available tools, can't do any of that. They are stateless and frozen in time."*
> — [Isaac Hagoel, "Why LLM Memory Still Fails"](https://dev.to/isaachagoel/why-llm-memory-still-fails-a-field-guide-for-builders-3d78)

> *"Having memory infrastructure doesn't solve the problem. Fine-tuning causes catastrophic forgetting. MemoryLLM, despite clever design with dedicated memory regions, failed on realistic conversational data, working only with unrealistic benchmark setups."*
> — Ibid.

Claude Code's own documentation [acknowledges the issue](https://code.claude.com/docs/en/hooks): hooks provide "deterministic control" but can't replicate cognitive functions. The built-in auto-memory feature (`~/.claude/` memory files) addresses this at the harness level — but it's a persistence mechanism, not a learning system.

[Gartner warns](https://www.personize.ai/blog/why-agents-fail-without-memory) that over 40% of agentic AI projects risk cancellation by 2027 if memory architecture is not established early. The challenge isn't building memory infrastructure — it's getting agents to reliably use it.

### The Deeper Issue

The experiment reveals a category error in the project's design. We tried to solve a **mechanism problem** with **instructions**.

| What we needed | What we built |
|---|---|
| Automatic encoding (like human implicit memory) | Instructions telling the AI to encode ("please remember this") |
| Surprise-driven salience (like Titans) | Keyword pattern matching on user messages |
| Integrated memory management (like MemGPT's agent loop) | Side-channel hooks that can't observe the AI's reasoning |
| Feedback signals for encoding failures | Nothing — silent failure with no learning |
| Persistent motivation across turns | System prompt instructions that fade with context length |

The cognitive science parallel is apt: we built the hippocampus (storage infrastructure) but not the amygdala (what makes something worth remembering), the prefrontal cortex (executive control over memory operations), or the dopaminergic system (reinforcement signal for successful encoding/retrieval).

### What Would Actually Work

Based on this experiment and the research, here's what would be needed for external LLM memory to actually work:

**1. Memory as a first-class agent capability, not an add-on.**

The model's agent loop must include memory operations as core actions — on the same level as "write code" or "read file." This is what MemGPT/Letta does, and what Anthropic's Claude Code is moving toward with its built-in memory files. The memory system can't be a skill, a hook, or a CLAUDE.md instruction — it needs to be part of the inference loop itself.

**2. Post-response hooks or model-integrated memory.**

If the architecture must remain external, the critical missing hook is `PostResponse` — the ability to process the AI's *output* and the conversation *after* the AI has responded. This would allow a separate process (or the model itself, in a second pass) to evaluate what was learned and store it. Claude Code's `PostResponse` hook type doesn't exist yet, but it's the most impactful integration point.

Alternatively, model providers could expose a "memory callback" API: after generating a response, the model emits structured memory operations (store, update, forget) as metadata alongside the text response.

**3. Retrieval-triggered encoding.**

Instead of relying on the AI to proactively store memories, use failed retrievals as a learning signal. When the AI needs information it doesn't have ("I don't know the user's name"), that gap should trigger encoding of the answer when it becomes available. This inverts the problem: instead of "remember to store," it's "notice what you don't know."

**4. Real embedding models for actual semantic similarity.**

The hash-based embeddings mean that even when memories are stored, recall by similarity doesn't work semantically. With real embeddings, the system could retrieve relevant context without the AI needing to explicitly search — similar to how human memory retrieval is cue-dependent and often involuntary.

### Conclusion

**The experiment hypothesis was:** *Can we give a coding assistant persistent, useful memory by building external infrastructure (SQLite + vector storage + graph + hooks) and instructing the AI to use it?*

**The answer is no.** Not because the infrastructure doesn't work — it does. But because current LLMs cannot be reliably instructed to perform ambient background tasks alongside their primary function. The instruction "manage your memory" competes with "help the user with their task" and consistently loses.

This is not a failure of prompting skill. It's a fundamental limitation of the instruction-following paradigm for cognitive functions. Memory encoding in humans is largely *unconscious and automatic* — driven by emotional salience, surprise, and repetition, not by a conscious decision to remember. Asking an LLM to consciously manage memory via instructions is like asking a human to consciously control their heartbeat: the mechanism exists at a different level than conscious intention.

The path forward is not better prompts or more aggressive instructions. It's architectural:
- **Model-level:** Memory mechanisms built into the architecture (Titans, Infini-attention)
- **Platform-level:** Memory as a first-class agent capability with feedback loops (Letta, Mem0)
- **Integration-level:** Post-response processing with real semantic embeddings, where encoding is triggered by salience signals rather than instructions

The infrastructure built here — tiered storage, decay-based forgetting, graph relations, consolidation — remains sound. The storage side of the problem is largely solved. What's missing is the *encoding* side: the mechanism that decides what to store, when, and why. That's the hard problem, and it can't be solved with a CLAUDE.md file.

## Where This Could Go

If this experiment were to continue, the most impactful next steps would be:

1. **Real embeddings.** Integrate a local embedding model (e.g., `nomic-embed-text` via `sentence-transformers`) so that similarity search actually works semantically. This is the single biggest upgrade.

2. **LLM-driven consolidation.** Periodically ask the LLM to review episodic memories and produce semantic summaries. "You've had 12 conversations about the auth system. Here's what you know: [consolidated summary]."

3. **Surprise-based importance.** Borrow from Titans: estimate the "surprise" of new information relative to existing memories. If the user tells you something that contradicts or significantly extends your memory, it's important. If it's redundant, skip it.

4. **Memory reflection.** Like the Generative Agents paper — periodically generate higher-level insights from accumulated memories. "Based on 30 sessions, this user prefers pragmatic solutions over elegant abstractions."

5. **Standardized memory protocol.** Define a simple, reusable API that any LLM integration can use. Store, recall, forget, relate, consolidate — as a protocol, not just a Python library.

## References

### Foundational Cognitive Science
- Atkinson, R.C. & Shiffrin, R.M. (1968). *Human memory: A proposed system and its control processes.* Psychology of Learning and Motivation.
- Baddeley, A.D. & Hitch, G. (1974). *Working memory.* Psychology of Learning and Motivation.
- Tulving, E. (1972). *Episodic and semantic memory.* Organization of Memory.
- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology.*

### Architectural Memory (In-Model)
- Behrouz, A., Zhong, P. & Mirrokni, V. (2024). [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663). NeurIPS 2025.
- Behrouz, A., Razaviyayn, M. & Mirrokni, V. (2025). [MIRAS: Unified Framework for Memory Architectures](https://arxiv.org/abs/2504.13173). Google Research.
- Wu, Y. et al. (2022). [Memorizing Transformers](https://arxiv.org/abs/2203.08913). ICLR 2022.
- Munkhdalai, T. et al. (2024). [Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143). Google.

### External Memory and RAG
- Lewis, P. et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS 2020.
- Qian, C. et al. (2024). [MemoRAG: Moving Towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery](https://arxiv.org/abs/2409.05591).

### Agent Memory Systems
- Packer, C. et al. (2023). [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560).
- Chheda, T. et al. (2025). [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413).
- Xu, W. et al. (2025). [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110). NeurIPS 2025.
- Park, J.S. et al. (2023). [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). UIST 2023.

### LLM Reliability and Instruction Following
- Li, T. et al. (2025). [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657). MAST taxonomy with 14 failure modes across 1,600+ traces.
- Wallace, E. et al. (2024). [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208).
- Hagoel, I. (2025). [Why LLM Memory Still Fails — A Field Guide for Builders](https://dev.to/isaachagoel/why-llm-memory-still-fails-a-field-guide-for-builders-3d78).

### Surveys and Overviews
- Liu, S. et al. (2025). [Memory in the Age of AI Agents: A Survey](https://arxiv.org/abs/2512.13564).
- Sun, H. et al. (2025). [From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs](https://arxiv.org/abs/2504.15965).
- ICLR 2026 Workshop Proposal. [MemAgents: Memory for LLM-Based Agentic Systems](https://openreview.net/pdf?id=U51WxL382H).
