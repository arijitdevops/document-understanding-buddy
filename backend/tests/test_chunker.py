from app.ingestion.chunker import chunk_pages, detect_heading, split_text
from app.ingestion.loaders import ParsedPage


def test_split_text_respects_size_and_keeps_all_words():
    text = " ".join(f"word{i}" for i in range(500))
    pieces = split_text(text, size=200, overlap=20)
    assert all(len(piece) <= 200 for piece in pieces)
    joined = " ".join(pieces)
    assert all(f"word{i}" in joined for i in range(500))


def test_split_text_prefers_paragraph_boundaries():
    text = "First paragraph sentence.\n\nSecond paragraph sentence."
    assert split_text(text, size=30, overlap=0) == [
        "First paragraph sentence.",
        "Second paragraph sentence.",
    ]


def test_detect_heading_variants():
    assert detect_heading("## Pricing") == (2, "Pricing")
    assert detect_heading("2.1 Scope of Work") == (2, "2.1 Scope of Work")
    assert detect_heading("TERMS AND CONDITIONS") == (1, "Terms And Conditions")
    assert detect_heading("An ordinary sentence.") is None


def test_chunk_pages_tracks_page_and_heading():
    pages = [
        ParsedPage(1, "# Intro\n" + "Alpha text. " * 30),
        ParsedPage(2, "# Pricing\n## Enterprise\n" + "Beta text. " * 30),
    ]
    chunks = chunk_pages(pages, size=200, overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert any(c.heading == "Pricing > Enterprise" for c in chunks if c.page == 2)
    assert all(c.token_count > 0 for c in chunks)
