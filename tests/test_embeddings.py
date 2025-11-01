from __future__ import annotations

import numpy as np
import pytest

from app.index.embeddings import normalize_embeddings


def test_normalize_embeddings_returns_unit_vectors() -> None:
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)
    assert pytest.approx(np.linalg.norm(normalized[0])) == 1.0
    # Zero vector should remain zero after normalization guard
    assert np.allclose(normalized[1], np.zeros(2))


def test_normalize_embeddings_requires_2d() -> None:
    with pytest.raises(ValueError):
        normalize_embeddings(np.array([1.0, 2.0]))
