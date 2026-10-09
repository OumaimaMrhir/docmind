import pytest

from app.chunking import chunk_text


def test_empty_text_gives_no_chunk():
    assert chunk_text("   ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("Une phrase courte.", chunk_size=100, overlap=10) == ["Une phrase courte."]


def test_chunks_respect_max_size():
    text = "\n\n".join(f"Paragraphe {i}. " + "mot " * 40 for i in range(10))
    chunks = chunk_text(text, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("texte", chunk_size=100, overlap=100)
