"""Index-related tooling for the ONH Expert Connector backend."""

from __future__ import annotations

from .chunking import Chunker, SimpleTokenizer, normalize_text
from .models import Chunk, IndexPaths, NvaPublicationSnippet, SourceLink, StaffRecord
from .records_loader import load_curated_records
from .embedder_factory import create_embedding_backend
from .vector_store import LocalVectorStore, SearchResult

__all__ = [
    "Chunk",
    "Chunker",
    "IndexPaths",
    "LocalVectorStore",
    "NvaPublicationSnippet",
    "SearchResult",
    "SimpleTokenizer",
    "SourceLink",
    "StaffRecord",
    "create_embedding_backend",
    "load_curated_records",
    "normalize_text",
]
