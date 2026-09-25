import pytest

from app.ingestion.loaders import UnsupportedDocumentError, load_document
from tests.helpers import make_docx, make_pdf


async def test_pdf_pages_are_numbered(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(make_pdf(["Revenue grew in March.", "Costs fell in April."]))
    pages = await load_document(path, "application/pdf")
    assert [p.page_number for p in pages] == [1, 2]
    assert "Revenue" in pages[0].text and "April" in pages[1].text


async def test_docx_headings_become_markdown(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(make_docx("Overview", ["Photosynthesis converts light.", "It needs water."]))
    pages = await load_document(path, "application/octet-stream")
    assert pages[0].text.startswith("# Overview")
    assert "Photosynthesis" in pages[0].text


async def test_csv_is_rendered_with_header(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,role\nAda,Engineer\nGrace,Admiral\n", encoding="utf-8")
    pages = await load_document(path, "text/csv")
    assert pages[0].text.splitlines()[0] == "name | role"
    assert "Grace | Admiral" in pages[0].text


async def test_markdown_and_text(tmp_path):
    md = tmp_path / "readme.md"
    md.write_text("# Title\n\nBody text", encoding="utf-8")
    txt = tmp_path / "plain.txt"
    txt.write_text("Just text", encoding="utf-8")
    assert (await load_document(md))[0].text.startswith("# Title")
    assert (await load_document(txt))[0].text == "Just text"


async def test_unsupported_extension(tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK")
    with pytest.raises(UnsupportedDocumentError):
        await load_document(path, "application/zip")
