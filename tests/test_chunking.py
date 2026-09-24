from app.ingestion.chunkers import chunk_markdown, chunk_plain_text, chunk_python, count_tokens
from app.ingestion.extractors import PageText, _group_into_blocks
from app.ingestion.chunkers import chunk_pdf
from tests.conftest import SAMPLES


def test_python_chunks_are_symbol_aligned_with_line_numbers():
    source = (SAMPLES / "Source_Code_Sample.py").read_text()
    chunks = chunk_python(source, "Source_Code_Sample.py", 350, 50)
    sections = [c.section for c in chunks]

    for symbol in [
        "DecayProxyRotator",
        "DecayProxyRotator.get_proxy",
        "DecayProxyRotator.report_failure",
        "UAFreshnessRotator.report_block",
        "run_test",
    ]:
        assert symbol in sections

    lines = source.splitlines()
    get_proxy = next(c for c in chunks if c.section == "DecayProxyRotator.get_proxy")
    assert lines[get_proxy.start_line - 1].strip().startswith("def get_proxy")
    assert "Defined in: class DecayProxyRotator" in get_proxy.embed_text

    # The comment banner above run_test stays attached to it rather than being orphaned.
    run_test = next(c for c in chunks if c.section == "run_test")
    assert run_test.content.startswith("# --- TEST SCRIPT ---")
    assert all(c.token_count <= 350 for c in chunks)


def test_invalid_python_falls_back_to_line_windows():
    chunks = chunk_python("def broken(:\n    pass\n", "bad.py", 350, 50)
    assert len(chunks) == 1 and chunks[0].start_line == 1


def test_text_chunks_respect_size_and_overlap():
    para = " ".join(f"sentence{i} is here." for i in range(400))
    chunks = chunk_plain_text(para, "t.txt", 100, 20)
    assert len(chunks) > 3
    assert all(count_tokens(c.content) <= 100 for c in chunks)
    # consecutive chunks share overlapping text
    tail = chunks[0].content.split()[-3:]
    assert " ".join(tail) in chunks[1].content


def test_markdown_chunks_keep_heading_path():
    md = "# Guide\nintro\n\n## Install\nrun pip install\n\n```\n# not a heading\n```\n\n## Usage\ncall it"
    chunks = chunk_markdown(md, "g.md", 350, 50)
    assert [c.section for c in chunks] == ["Guide", "Guide > Install", "Guide > Usage"]
    assert "# not a heading" in chunks[1].content


def test_pdf_chunks_track_pages():
    pages = [PageText(1, "alpha " * 200), PageText(2, "beta " * 200, ocr=True)]
    chunks = chunk_pdf(pages, "d.pdf", 150, 20)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert chunks[-1].metadata.get("ocr") is True


def test_ocr_block_grouping_separates_columns():
    # two columns of lines at the same heights must not be interleaved
    lines = [
        (10, 0, 300, 20, "left one"), (400, 0, 700, 20, "right one"),
        (10, 25, 300, 45, "left two"), (400, 25, 700, 45, "right two"),
    ]
    assert _group_into_blocks(lines) == ["left one left two", "right one right two"]


def test_repeated_footers_are_stripped_despite_ocr_variants():
    from app.ingestion.extractors import _strip_repeated_headers_footers

    bodies = ["Agents reason.", "MCP is a standard.", "Remote scaled IT.", "Phase three."]
    pages = [f"{b}\nzapier\nPage {i}" for i, b in enumerate(bodies, 1)]
    pages[1] = pages[1].replace("zapier", "_zapier")
    assert _strip_repeated_headers_footers(pages) == bodies
