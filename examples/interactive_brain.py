#!/usr/bin/env python3
"""Interactive brain usage - create memories and watch the dashboard update."""

import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_brain import Brain, MemoryTier


def create_test_vector(text: str, dim: int = 128) -> np.ndarray:
    """Create a deterministic vector from text for testing."""
    # Use hash to make similar texts have similar vectors
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)


def main():
    """Interactive brain session."""
    print("🧠 LLM-Brain Interactive Session")
    print("=" * 50)
    print()

    # Create a temporary brain for testing
    # (Or use default: brain = Brain())
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Using brain at: {tmpdir}")
        brain = Brain(brain_path=tmpdir, vector_dimensions=128)

        health = brain.health_check()
        print(f"✅ Brain initialized: {health['initialized']}")
        print(f"✅ Writable: {health['writable']}")
        print()

        print("💡 Tips:")
        print("   - Watch the dashboard at http://localhost:8080")
        print("   - Each operation will appear in the activity feed")
        print("   - Memories are ranked by importance")
        print()

        memory_ids = []

        while True:
            print()
            print("Commands:")
            print("  1. Create memory")
            print("  2. Recall by text similarity")
            print("  3. Recall important memories")
            print("  4. Show all memories")
            print("  5. Create relation between memories")
            print("  6. Forget memory")
            print("  7. Show stats")
            print("  q. Quit")
            print()

            choice = input("> ").strip().lower()

            if choice == "1":
                text = input("Enter memory text: ").strip()
                if not text:
                    print("⚠️  Empty text, skipping")
                    continue

                importance = float(input("Importance (0.0-1.0, default 0.5): ") or "0.5")
                tier = (
                    input("Tier (working/episodic/semantic, default working): ").strip()
                    or "working"
                )

                vector = create_test_vector(text)
                memory_id = brain.memorize(
                    vector=vector,
                    text=text,
                    importance=importance,
                    tier=tier,
                )
                memory_ids.append(memory_id)
                print(f"✅ Created memory: {memory_id[:8]}...")
                print(f"   Check the dashboard - it should appear in recent memories!")

            elif choice == "2":
                query = input("Search query: ").strip()
                if not query:
                    print("⚠️  Empty query")
                    continue

                query_vector = create_test_vector(query)
                results = brain.recall(query_vector=query_vector, top_k=5)

                print(f"\n🔍 Found {len(results)} similar memories:")
                for memory, similarity in results:
                    print(f"  • [{memory.tier.value}] {memory.raw_text[:50]}...")
                    print(
                        f"    Similarity: {similarity:.3f} | Importance: {memory.metadata.importance_score:.2f}"
                    )

            elif choice == "3":
                tier = input("Which tier? (working/episodic/semantic, default all): ").strip()

                if tier:
                    memories = brain.recall_important(MemoryTier(tier), top_k=5)
                    print(f"\n⭐ Most important in {tier}:")
                else:
                    # Get from all tiers
                    memories = []
                    for t in MemoryTier:
                        memories.extend(brain.recall_important(t, top_k=3))
                    memories.sort(key=lambda m: m.metadata.importance_score, reverse=True)
                    print(f"\n⭐ Most important memories:")

                for memory in memories[:5]:
                    print(f"  • [{memory.tier.value}] {memory.raw_text[:50]}...")
                    print(
                        f"    Importance: {memory.metadata.importance_score:.2f} | Accessed: {memory.metadata.access_count}x"
                    )

            elif choice == "4":
                print("\n📚 All memories:")
                for tier in MemoryTier:
                    memories = brain.recall_by_tier(tier)
                    print(f"\n  {tier.value.upper()} ({len(memories)}):")
                    for memory in memories[:3]:  # Show first 3
                        print(
                            f"    • {memory.raw_text[:40]}... (⭐ {memory.metadata.importance_score:.2f})"
                        )
                    if len(memories) > 3:
                        print(f"    ... and {len(memories) - 3} more")

            elif choice == "5":
                if len(memory_ids) < 2:
                    print("⚠️  Need at least 2 memories. Create some first!")
                    continue

                print(f"Available memories: {len(memory_ids)}")
                for i, mid in enumerate(memory_ids[:5]):
                    print(f"  {i}: {mid[:8]}...")

                try:
                    idx1 = int(input("First memory index: "))
                    idx2 = int(input("Second memory index: "))
                    relation = input("Relation type (default 'related'): ").strip() or "related"

                    brain.relate(memory_ids[idx1], memory_ids[idx2], relation)
                    print(
                        f"✅ Created relation: {memory_ids[idx1][:8]}... --[{relation}]--> {memory_ids[idx2][:8]}..."
                    )
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Invalid selection: {e}")

            elif choice == "6":
                if not memory_ids:
                    print("⚠️  No memories to forget")
                    continue

                memory_id = input("Memory ID to forget (or 'last' for most recent): ").strip()
                if memory_id == "last":
                    memory_id = memory_ids[-1]

                if brain.forget(memory_id):
                    print(f"✅ Forgot memory: {memory_id[:8]}...")
                    if memory_id in memory_ids:
                        memory_ids.remove(memory_id)
                else:
                    print("❌ Memory not found")

            elif choice == "7":
                stats = brain.stats()
                print("\n📊 Brain Statistics:")
                print(f"  Tiers: {stats['tiers']}")
                print(f"  Graph available: {stats['graph']['available']}")
                print(f"  Brain path: {stats['config']['brain_path']}")

            elif choice in ("q", "quit", "exit"):
                print("👋 Goodbye!")
                break

            else:
                print("Unknown command")


if __name__ == "__main__":
    main()
