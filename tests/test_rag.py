from app.ingest import chunk_text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_single_chunk_when_short():
    text = "one two three four five"
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == [text]


def test_chunk_text_splits_long_input():
    words = [f"word{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) > 1
    # every chunk should be non-empty and within the requested size
    for chunk in chunks:
        assert chunk
        assert len(chunk.split()) <= 500


def test_chunk_text_overlap_shares_words():
    words = [f"word{i}" for i in range(700)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    first_tail = chunks[0].split()[-100:]
    second_head = chunks[1].split()[:100]
    assert first_tail == second_head
