"""CLI wrapper that automatically uses brain for all interactions.

This provides a drop-in replacement for CLI usage that automatically:
- Loads memories at session start
- Recalls relevant context before each prompt
- Stores exchanges after each response

Usage:
    llm-brain-chat  # Instead of normal chat CLI
"""

from __future__ import annotations

import numpy as np

from llm_brain import Brain
from llm_brain.memory.models import MemoryTier


class BrainChatSession:
    """Interactive chat session with automatic brain integration."""

    def __init__(
        self,
        brain_path: str | None = None,
        vector_dimensions: int = 128,
    ) -> None:
        """Initialize chat session with brain."""
        self.brain = Brain(
            brain_path=brain_path,
            vector_dimensions=vector_dimensions,
        )
        self.session_memories: list[str] = []

    def _embed(self, text: str) -> np.ndarray:
        """Create embedding."""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self.brain.config.vector_dimensions).astype(np.float32)

    def start(self) -> None:
        """Start the chat session."""
        print("🧠 LLM-Brain Chat Session")
        print("=" * 50)

        # Load existing memories
        memories = self.brain.recall_important(MemoryTier.WORKING, top_k=5)
        if memories:
            print("\n📚 Loaded context:")
            for m in memories:
                print(f"  • {m.raw_text[:70]}...")
            print()

        print("Type 'exit' to quit, 'memories' to see all memories\n")

    def process_input(self, user_input: str) -> str:
        """Process user input with automatic recall and store."""
        # Handle special commands
        if user_input.lower() == "exit":
            return "__EXIT__"

        if user_input.lower() == "memories":
            self._show_memories()
            return ""

        # RECALL: Search for relevant context
        query_vector = self._embed(user_input)
        relevant = self.brain.recall(query_vector=query_vector, top_k=3)

        context = ""
        if relevant:
            context = "\n[Relevant context from memory]:\n"
            for memory, _similarity in relevant:
                context += f"  - {memory.raw_text[:80]}...\n"
            context += "\n"

        # Here you would normally call the LLM with context
        # For now, just echo with context
        response = f"{context}[Response to: {user_input[:50]}...]"

        # STORE: Save the exchange
        exchange = f"User: {user_input}\nAssistant: {response}"
        mid = self.brain.memorize(
            vector=self._embed(exchange),
            text=exchange,
            importance=self._calculate_importance(user_input),
        )
        self.session_memories.append(mid)

        return response

    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for text."""
        # Simple heuristic
        importance = 0.5  # Base

        # Increase for preferences, facts, decisions
        indicators = [
            "prefer",
            "like",
            "want",
            "need",
            "decided",
            "important",
            "remember",
        ]
        for indicator in indicators:
            if indicator in text.lower():
                importance += 0.2

        # Cap at 0.95
        return min(importance, 0.95)

    def _show_memories(self) -> None:
        """Display all memories."""
        print("\n📚 All Memories:")
        for tier in MemoryTier:
            memories = self.brain.recall_by_tier(tier)
            if memories:
                print(f"\n  {tier.value.upper()} ({len(memories)}):")
                for m in memories[:5]:
                    print(f"    • {m.raw_text[:60]}... (⭐ {m.metadata.importance_score:.2f})")
                if len(memories) > 5:
                    print(f"    ... and {len(memories) - 5} more")
        print()

    def end(self) -> None:
        """End session and store summary."""
        # Store session summary
        if self.session_memories:
            summary = f"Chat session with {len(self.session_memories)} exchanges"
            self.brain.memorize(
                vector=self._embed(summary),
                text=summary,
                importance=0.6,
            )

        print(f"\n💾 Stored {len(self.session_memories)} memories")
        self.brain.close()


def main() -> None:
    """Main entry point for brain-integrated chat."""
    session = BrainChatSession()
    session.start()

    try:
        while True:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue

                response = session.process_input(user_input)

                if response == "__EXIT__":
                    break

                if response:
                    print(response)

            except KeyboardInterrupt:
                print("\n")
                break

    finally:
        session.end()


if __name__ == "__main__":
    main()
