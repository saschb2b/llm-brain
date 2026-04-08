"""Basic usage example for LLM-Brain."""

import numpy as np

from llm_brain import Brain, MemoryTier


def main():
    """Demonstrate basic LLM-Brain functionality."""

    print("=" * 60)
    print("LLM-Brain Basic Usage Example")
    print("=" * 60)

    # Initialize brain
    print("\n1. Initializing brain...")
    brain = Brain(vector_dimensions=128)  # Small for demo
    health = brain.health_check()
    print(f"   ✓ Brain initialized: {health['initialized']}")
    print(f"   ✓ Writable: {health['writable']}")

    # Store memories
    print("\n2. Storing memories...")

    # Memory 1: User preference
    vector1 = np.random.randn(128).astype(np.float32)
    id1 = brain.memorize(
        vector=vector1,
        text="User prefers Python for data processing",
        importance=0.9,
        tier=MemoryTier.WORKING,
    )
    print(f"   ✓ Stored memory 1: {id1[:8]}...")

    # Memory 2: Important fact
    vector2 = np.random.randn(128).astype(np.float32)
    id2 = brain.memorize(
        vector=vector2,
        text="Important project deadline is Friday",
        importance=0.95,
        tier=MemoryTier.WORKING,
    )
    print(f"   ✓ Stored memory 2: {id2[:8]}...")

    # Memory 3: Low importance
    vector3 = np.random.randn(128).astype(np.float32)
    id3 = brain.memorize(
        vector=vector3, text="Temporary note about syntax", importance=0.3, tier=MemoryTier.WORKING
    )
    print(f"   ✓ Stored memory 3: {id3[:8]}...")

    # Create relation
    print("\n3. Creating relation...")
    brain.relate(id1, id2, "extends", weight=0.8)
    print(f"   ✓ Related memory 1 → memory 2 (extends)")

    # Recall by ID
    print("\n4. Recalling by ID...")
    memory = brain.recall(memory_id=id1)
    if memory:
        print(f"   ✓ Found: {memory.raw_text}")
        print(f"   ✓ Importance: {memory.metadata.importance_score}")

    # Recall by similarity
    print("\n5. Similarity search...")
    query = vector1 * 0.95 + np.random.randn(128) * 0.05  # Similar to vector1
    results = brain.recall(query_vector=query, top_k=3)
    print(f"   ✓ Found {len(results)} similar memories:")
    for mem, score in results:
        print(f"      - {mem.raw_text[:40]}... (score: {score:.3f})")

    # Recall important memories
    print("\n6. Most important memories...")
    important = brain.recall_important(MemoryTier.WORKING, top_k=3)
    print(f"   ✓ Top {len(important)} memories by importance:")
    for mem in important:
        print(f"      - {mem.raw_text[:40]}... (importance: {mem.metadata.importance_score})")

    # Get related memories
    print("\n7. Related memories...")
    related = brain.recall_related(id1)
    print(f"   ✓ Found {len(related)} related memories:")
    for rel in related:
        print(
            f"      - {rel['memory_id'][:8]}... ({rel['relation_type']}, weight: {rel['weight']})"
        )

    # Tier statistics
    print("\n8. Tier statistics...")
    stats = brain.tier_manager.get_tier_stats()
    for tier, stat in stats.items():
        print(f"   {tier}: {stat['count']} memories")

    # Consolidate
    print("\n9. Running consolidation...")
    consolidation = brain.consolidate()
    print(f"   ✓ Promoted: {consolidation['promoted']}")
    print(f"   ✓ Evicted from working: {consolidation['working_evicted']}")

    # Final stats
    print("\n10. Final brain statistics...")
    final_stats = brain.stats()
    print(f"    Config: {final_stats['config']['brain_path']}")
    print(f"    Graph available: {final_stats['graph']['available']}")

    # Cleanup
    print("\n11. Cleanup...")
    brain.forget(id1)
    brain.forget(id2)
    brain.forget(id3)
    print("   ✓ Deleted test memories")

    brain.close()
    print("   ✓ Brain closed")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
