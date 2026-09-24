"""Chunking strategies.

* Prose (PDF / text): recursive split on paragraph -> line -> sentence -> word boundaries,
  greedily packed to ~``max_tokens`` with ``overlap`` tokens carried between chunks.
  PDF chunks track the page range they came from.
* Markdown: split by heading first so a chunk never straddles two sections; the heading
  path is kept as the chunk's ``section`` and prepended to the embedded text.
* Python: AST-aware. One chunk per function / method (with leading comments and
  decorators), a class-overview chunk, and chunks for remaining module-level code.
  Line numbers are preserved so results can be cited as ``file.py:L42-L60``.
* Other code: line-window packing that never splits inside a line.

Each chunk has ``content`` (what users see) and ``embed_text`` (what gets embedded):
``embed_text`` adds a small context header (file name, symbol / heading) so short chunks
still carry the context they need to be found.
"""

import ast
import re
from dataclasses import dataclass, field

from app.ingestion.extractors import PageText

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Cheap, tokenizer-free estimate. WordPiece produces somewhat more tokens than this,
    which is why the default chunk size (350) leaves headroom under bge's 512 limit."""
    return len(_TOKEN_RE.findall(text))


@dataclass
class ChunkDraft:
    content: str
    embed_text: str
    chunk_type: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return count_tokens(self.content)


# ---------------------------------------------------------------------------
# Generic recursive splitting
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _atomize(text: str, max_tokens: int, seps: list[str] = _SEPARATORS) -> list[str]:
    """Split text into ordered pieces each <= max_tokens, preferring coarse boundaries.
    Separators stay attached to the preceding piece so ''.join(pieces) == text."""
    if count_tokens(text) <= max_tokens or not seps:
        return [text]
    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _atomize(text, max_tokens, rest)
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if not piece:
            continue
        if count_tokens(piece) > max_tokens:
            pieces.extend(_atomize(piece, max_tokens, rest))
        else:
            pieces.append(piece)
    return pieces


def _pack(atoms: list[tuple[str, int | None]], max_tokens: int, overlap: int):
    """Greedily pack (text, page) atoms into windows with token overlap.
    Yields (text, page_start, page_end)."""
    window: list[tuple[str, int | None, int]] = []
    size = 0
    for text, page in atoms:
        n = count_tokens(text)
        if window and size + n > max_tokens:
            yield _emit(window)
            # carry the tail of the previous window forward as overlap
            carry: list[tuple[str, int | None, int]] = []
            carried = 0
            for item in reversed(window):
                if carried + item[2] > overlap:
                    break
                carry.insert(0, item)
                carried += item[2]
            window, size = carry, carried
        window.append((text, page, n))
        size += n
    if window:
        yield _emit(window)


def _emit(window):
    text = "".join(t for t, _, _ in window).strip()
    pages = [p for _, p, _ in window if p is not None]
    return text, (min(pages) if pages else None), (max(pages) if pages else None)


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------


def _paragraphs(text: str) -> list[str]:
    """Rebuild paragraphs from extracted text: blank lines separate paragraphs and
    single newlines inside a paragraph become spaces (PDF line wrapping)."""
    paras = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if lines:
            paras.append(" ".join(lines))
    return paras


def chunk_pdf(pages: list[PageText], filename: str, max_tokens: int, overlap: int) -> list[ChunkDraft]:
    atoms: list[tuple[str, int | None]] = []
    for page in pages:
        for para in _paragraphs(page.text):
            for piece in _atomize(para + "\n\n", max_tokens):
                atoms.append((piece, page.page))
    ocr_pages = {p.page for p in pages if p.ocr}
    chunks = []
    for text, p0, p1 in _pack(atoms, max_tokens, overlap):
        if not text:
            continue
        pages_label = f"page {p0}" if p0 == p1 else f"pages {p0}-{p1}"
        chunks.append(
            ChunkDraft(
                content=text,
                embed_text=f"Document: {filename} ({pages_label})\n\n{text}",
                chunk_type="text",
                page_start=p0,
                page_end=p1,
                metadata={"ocr": True} if ocr_pages & set(range(p0, p1 + 1)) else {},
            )
        )
    return chunks


def chunk_plain_text(text: str, filename: str, max_tokens: int, overlap: int) -> list[ChunkDraft]:
    atoms = [(p, None) for para in _paragraphs(text) for p in _atomize(para + "\n\n", max_tokens)]
    return [
        ChunkDraft(content=t, embed_text=f"Document: {filename}\n\n{t}", chunk_type="text")
        for t, _, _ in _pack(atoms, max_tokens, overlap)
        if t
    ]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def chunk_markdown(text: str, filename: str, max_tokens: int, overlap: int) -> list[ChunkDraft]:
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    path: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            level, title = len(match.group(1)), match.group(2)
            path = [(l, t) for l, t in path if l < level] + [(level, title)]
            sections.append((" > ".join(t for _, t in path), [line]))
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        # Code fences are kept intact where possible by splitting on blank lines first.
        atoms = [(p, None) for p in _atomize(body, max_tokens)]
        for t, _, _ in _pack(atoms, max_tokens, overlap):
            if not t:
                continue
            header = f"Document: {filename}" + (f"\nSection: {heading}" if heading else "")
            chunks.append(
                ChunkDraft(content=t, embed_text=f"{header}\n\n{t}", chunk_type="markdown", section=heading)
            )
    return chunks


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


def _pack_lines(lines: list[str], first_line: int, max_tokens: int, overlap: int):
    """Pack source lines into windows. Yields (text, start_line, end_line), 1-based."""
    window: list[tuple[int, str, int]] = []
    size = 0
    for offset, line in enumerate(lines):
        n = count_tokens(line) + 1
        if window and size + n > max_tokens:
            yield _emit_lines(window)
            carry, carried = [], 0
            for item in reversed(window):
                if carried + item[2] > overlap:
                    break
                carry.insert(0, item)
                carried += item[2]
            window, size = carry, carried
        window.append((first_line + offset, line, n))
        size += n
    if window:
        yield _emit_lines(window)


def _emit_lines(window):
    return "\n".join(l for _, l, _ in window), window[0][0], window[-1][0]


def _code_chunks_for_block(
    source_lines: list[str],
    start: int,
    end: int,
    *,
    filename: str,
    language: str,
    symbol: str | None,
    kind: str,
    context: str | None,
    max_tokens: int,
    overlap: int,
) -> list[ChunkDraft]:
    """Turn lines [start, end] (1-based, inclusive) into one or more code chunks."""
    lines = source_lines[start - 1 : end]
    while lines and not lines[0].strip():
        lines, start = lines[1:], start + 1
    while lines and not lines[-1].strip():
        lines, end = lines[:-1], end - 1
    if not lines:
        return []
    header = [f"File: {filename}", f"Language: {language}"]
    if symbol:
        header.append(f"Symbol: {symbol} ({kind})")
    if context:
        header.append(context)
    header_text = "\n".join(header)

    out = []
    for text, s, e in _pack_lines(lines, start, max_tokens, overlap):
        if not text.strip():
            continue
        out.append(
            ChunkDraft(
                content=text,
                embed_text=f"{header_text}\n\n{text}",
                chunk_type="code",
                section=symbol or "module",
                start_line=s,
                end_line=e,
                metadata={"symbol_kind": kind},
            )
        )
    return out


def _leading_comment_start(lines: list[str], node_start: int) -> int:
    """Walk upwards from a definition (1-based line) over comment and blank lines so the
    comment block above it (e.g. ``# --- TEST SCRIPT ---``) stays with the code."""
    start = node_start
    i = node_start - 2  # 0-based index of the line above the definition
    while i >= 0 and (not lines[i].strip() or lines[i].strip().startswith("#")):
        if lines[i].strip():
            start = i + 1
        i -= 1
    return start


def chunk_python(source: str, filename: str, max_tokens: int, overlap: int) -> list[ChunkDraft]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_code_generic(source, filename, "python", max_tokens, overlap)

    lines = source.splitlines()
    covered: set[int] = set()
    chunks: list[ChunkDraft] = []
    common = dict(filename=filename, language="python", max_tokens=max_tokens, overlap=overlap)

    def node_span(node) -> tuple[int, int]:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        return _leading_comment_start(lines, first), node.end_lineno

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e = node_span(node)
            covered.update(range(s, e + 1))
            chunks += _code_chunks_for_block(
                lines, s, e, symbol=node.name, kind="function", context=None, **common
            )
        elif isinstance(node, ast.ClassDef):
            s, e = node_span(node)
            covered.update(range(s, e + 1))
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            class_sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            doc = ast.get_docstring(node)

            # Class overview: signature, docstring, class-level statements and method list.
            method_lines = {ln for m in methods for ln in range(node_span(m)[0], m.end_lineno + 1)}
            body_lines = [
                lines[ln - 1] for ln in range(s, e + 1) if ln not in method_lines and lines[ln - 1].strip()
            ]
            overview = "\n".join(body_lines)
            if methods:
                sigs = [f"    def {m.name}({ast.unparse(m.args)})" for m in methods]
                overview += "\n    # methods:\n" + "\n".join(sigs)
            chunks.append(
                ChunkDraft(
                    content=overview,
                    embed_text=(
                        f"File: {filename}\nLanguage: python\nSymbol: {node.name} (class)\n"
                        + (f"Docstring: {doc}\n" if doc else "")
                        + f"\n{overview}"
                    ),
                    chunk_type="code",
                    section=node.name,
                    start_line=s,
                    end_line=e,
                    metadata={"symbol_kind": "class", "methods": [m.name for m in methods]},
                )
            )
            for m in methods:
                ms, me = node_span(m)
                chunks += _code_chunks_for_block(
                    lines,
                    ms,
                    me,
                    symbol=f"{node.name}.{m.name}",
                    kind="method",
                    context=f"Defined in: {class_sig}",
                    **common,
                )

    # Remaining module-level code (imports, constants, `if __name__ == "__main__":` ...).
    block: list[int] = []
    blocks: list[list[int]] = []
    for ln in range(1, len(lines) + 1):
        if ln in covered:
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(ln)
    if block:
        blocks.append(block)
    for b in blocks:
        if any(lines[ln - 1].strip() for ln in b):
            chunks += _code_chunks_for_block(
                lines, b[0], b[-1], symbol=None, kind="module", context=None, **common
            )

    chunks.sort(key=lambda c: (c.start_line or 0, 0 if c.metadata.get("symbol_kind") == "class" else 1))
    return chunks


def chunk_code_generic(
    source: str, filename: str, language: str, max_tokens: int, overlap: int
) -> list[ChunkDraft]:
    lines = source.splitlines()
    return _code_chunks_for_block(
        lines,
        1,
        len(lines),
        filename=filename,
        language=language,
        symbol=None,
        kind="module",
        context=None,
        max_tokens=max_tokens,
        overlap=overlap,
    )
