#!/usr/bin/env python3
"""Bootstrap script for LLM-Brain initialization.

This script can be run to initialize the brain storage and verify setup.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_brain import Brain


def main() -> int:
    """Main bootstrap function."""
    parser = argparse.ArgumentParser(description="Initialize LLM-Brain storage")
    parser.add_argument(
        "--path", type=str, default=None, help="Brain storage path (default: ~/.kimi-brain)"
    )
    parser.add_argument(
        "--dimensions", type=int, default=3072, help="Vector dimensions (default: 3072)"
    )
    parser.add_argument("--check", action="store_true", help="Only check health, don't initialize")
    parser.add_argument("--json", action="store_true", help="Output JSON format")

    args = parser.parse_args()

    # Create or check brain
    brain = Brain(brain_path=args.path, vector_dimensions=args.dimensions, auto_init=not args.check)

    if args.check:
        health = brain.health_check()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print("Health Check:")
            for key, value in health.items():
                print(f"  {key}: {value}")

        return 0 if health["initialized"] and health["writable"] else 1

    # Initialize if needed
    if not brain.db.is_initialized():
        brain.initialize()
        print(f"✓ Initialized brain at: {brain.config.brain_path}")
    else:
        print(f"✓ Brain already initialized at: {brain.config.brain_path}")

    # Run health check
    health = brain.health_check()

    if args.json:
        print(json.dumps(health, indent=2))
    else:
        print("\nHealth Status:")
        print(f"  Initialized: {health['initialized']}")
        print(f"  Schema Version: {health.get('schema_version', 'unknown')}")
        print(f"  SQLite-vec: {'✓' if health['sqlite_vec_loaded'] else '✗'}")
        print(f"  Graph DB: {'✓' if health['graph_available'] else '✗'}")
        print(f"  Writable: {'✓' if health['writable'] else '✗'}")

    brain.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
