from __future__ import annotations

import pytest
from app.index.chunking import Chunker, SimpleTokenizer, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    text = "  Dette  er  \n en  test.  "
    assert normalize_text(text) == "Dette er en test."


def test_chunker_splits_with_overlap() -> None:
    tokenizer = SimpleTokenizer()
    chunker = Chunker(chunk_size=5, chunk_overlap=2, tokenizer=tokenizer)
    words = " ".join(f"ord{i}" for i in range(12))
    chunks = chunker.chunk_text(
        staff_slug="slug", text=words, source_url="https://example.com"
    )

    assert len(chunks) == 4
    assert chunks[0].token_count == 5
    # Overlap ensures shared boundary tokens between consecutive chunks
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[-1].token_count == 3


def test_chunker_respects_max_chunks() -> None:
    chunker = Chunker(chunk_size=3, chunk_overlap=1, max_chunks=2)
    words = " ".join(str(i) for i in range(15))
    chunks = chunker.chunk_text(
        staff_slug="tester", text=words, source_url="https://example.com"
    )

    assert len(chunks) == 2
    assert all(chunk.staff_slug == "tester" for chunk in chunks)


def test_chunker_handles_empty_text() -> None:
    chunker = Chunker()
    assert (
        chunker.chunk_text(
            staff_slug="empty", text="   ", source_url="https://example.com"
        )
        == []
    )


def test_chunker_uses_source_namespace_and_offsets() -> None:
    chunker = Chunker(chunk_size=4, chunk_overlap=0)
    text = "en to tre fire fem seks"
    chunks = chunker.chunk_text(
        staff_slug="slug",
        text=text,
        source_url="https://example.com",
        source_namespace="nva",
        start_index=7,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "slug-nva-0007",
        "slug-nva-0008",
    ]
    assert [chunk.order for chunk in chunks] == [7, 8]


def test_chunker_respects_per_call_max_chunks_override() -> None:
    chunker = Chunker(chunk_size=3, chunk_overlap=1, max_chunks=5)
    words = " ".join(str(i) for i in range(15))
    chunks = chunker.chunk_text(
        staff_slug="tester",
        text=words,
        source_url="https://example.com",
        source_namespace="profile",
        max_chunks=2,
    )
    assert len(chunks) == 2


@pytest.mark.parametrize(
    "chunk_size, overlap", [(0, 0), (5, 5), (5, 6), (-1, 0), (5, -1)]
)
def test_chunker_validates_parameters(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        Chunker(chunk_size=chunk_size, chunk_overlap=overlap)


def test_chunker_validates_start_index() -> None:
    chunker = Chunker()
    with pytest.raises(ValueError):
        chunker.chunk_text(
            staff_slug="slug",
            text="hello",
            source_url="https://example.com",
            start_index=-1,
        )
