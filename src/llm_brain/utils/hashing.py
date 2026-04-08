"""Content hashing utilities for deduplication."""

import hashlib
from typing import Any, Optional

import numpy as np


def content_hash(
    text: Optional[str] = None,
    vector: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Generate SHA-256 hash for memory content.

    This enables deduplication by detecting identical or near-identical
    content before storage.

    Args:
        text: Raw text content
        vector: Embedding vector
        metadata: Additional metadata to include in hash

    Returns:
        Hex digest of hash
    """
    hasher = hashlib.sha256()

    if text:
        hasher.update(text.encode("utf-8"))

    if vector is not None:
        # Quantize vector for fuzzy matching
        quantized: np.ndarray = (vector * 1000).astype(np.int16)
        hasher.update(quantized.tobytes())

    if metadata:
        # Sort keys for consistent hashing
        for key in sorted(metadata.keys()):
            hasher.update(f"{key}:{metadata[key]}".encode())

    return hasher.hexdigest()


def vector_similarity_hash(vector: np.ndarray, buckets: int = 16) -> str:
    """Generate locality-sensitive hash for vectors.

    Useful for finding near-duplicate vectors quickly.

    Args:
        vector: Input vector
        buckets: Number of buckets for quantization

    Returns:
        Hash string representing vector region
    """
    # Simple LSH: divide vector space into buckets
    min_val, max_val = vector.min(), vector.max()
    bucket_size = (max_val - min_val) / buckets

    bucket_indices = ((vector - min_val) / bucket_size).astype(int)
    bucket_indices = np.clip(bucket_indices, 0, buckets - 1)

    return "".join(format(b, "x") for b in bucket_indices[:16])  # First 16 dims
