import pytest

from app.chunking import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("   \n\n  ") == []


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"This is sentence number {i}." for i in range(100))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "sentence number 0." in chunks[0]
    assert "sentence number 99." in chunks[-1]


def test_overlap_shares_content_between_neighbours():
    text = " ".join(f"Sentence {i} is here." for i in range(60))
    chunks = split_text(text, chunk_size=150, overlap=50)
    shared = [c for a, b in zip(chunks, chunks[1:]) for c in [a.split(". ")[-1]] if c and c in b]
    assert shared


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, overlap=100)
