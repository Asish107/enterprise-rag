from app.ingestion import chunk_text, load_document


def test_load_text(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Hello world.\n\nSecond paragraph.")
    text, fmt = load_document(p)
    assert fmt == "txt"
    assert "Hello world." in text
    assert "Second paragraph." in text


def test_load_markdown(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nBody text here.")
    text, fmt = load_document(p)
    assert fmt == "md"
    assert "Body text here." in text


def test_unsupported_format(tmp_path):
    p = tmp_path / "doc.xyz"
    p.write_text("nope")
    try:
        load_document(p)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_chunking_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunking_empty():
    assert chunk_text("   ") == []
