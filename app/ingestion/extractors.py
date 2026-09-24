"""Content extraction: bytes -> text units that keep their provenance (page numbers).

PDFs: the embedded text layer is used when present. Pages without one (scans, or - as in
the provided Knowledge_Base_Sample.pdf - text drawn as vector outlines) are rendered and
OCR'd. OCR lines are grouped into layout blocks so multi-column pages and call-out boxes
don't get interleaved line-by-line.
"""

import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass

from app.errors import ExtractionError

log = logging.getLogger(__name__)

MIN_TEXT_LAYER_CHARS = 25


@dataclass
class PageText:
    page: int  # 1-based
    text: str
    ocr: bool = False


class PdfOcr:
    """RapidOCR (PP-OCR models on ONNX Runtime): pip-installable, no system Tesseract.

    Two passes: detect + recognize at ``dpi``; then any line recognized with low
    confidence is re-recognized from a sharper ``rescue_dpi`` crop (cheap - only a few
    lines per page), which fixes most garbled lines at a fraction of full high-DPI cost.
    """

    def __init__(self, dpi: int = 150, rescue_dpi: int = 300, min_confidence: float = 0.5,
                 rescue_below: float = 0.85):
        self.dpi = dpi
        self.rescue_dpi = rescue_dpi
        self.min_confidence = min_confidence
        self.rescue_below = rescue_below
        self._engine = None
        self._lock = threading.Lock()

    def _get_engine(self):
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR(params={"Global.log_level": "warning"})
        return self._engine

    @staticmethod
    def _render(page, dpi: int):
        import numpy as np

        pix = page.get_pixmap(dpi=dpi)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]

    def page_text(self, page) -> str:
        import numpy as np

        with self._lock:  # one OCR at a time; each run already uses all cores
            engine = self._get_engine()
            # Flags are passed explicitly: RapidOCR remembers them between calls.
            result = engine(self._render(page, self.dpi), use_det=True, use_cls=True, use_rec=True)
            # When detection finds no text regions RapidOCR returns a recognition-only
            # result without boxes (e.g. an illustration-only page).
            if getattr(result, "boxes", None) is None or not result.txts:
                return ""
            lines = []
            hi_res = None
            scale = self.rescue_dpi / self.dpi
            for box, text, score in zip(result.boxes, result.txts, result.scores):
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                if score < self.rescue_below:
                    if hi_res is None:
                        hi_res = self._render(page, self.rescue_dpi)
                    pad = 6
                    crop = np.ascontiguousarray(
                        hi_res[max(0, int(y0 * scale) - pad) : int(y1 * scale) + pad,
                               max(0, int(x0 * scale) - pad) : int(x1 * scale) + pad]
                    )
                    retry = engine(crop, use_det=False, use_cls=False, use_rec=True)
                    if retry.txts and retry.scores[0] > score:
                        text, score = retry.txts[0], retry.scores[0]
                if score >= self.min_confidence and text.strip():
                    lines.append((x0, y0, x1, y1, text.strip()))
        return "\n\n".join(_group_into_blocks(lines))


def _group_into_blocks(lines: list[tuple[float, float, float, float, str]]) -> list[str]:
    """Group OCR line boxes into paragraphs: a line joins a block when it sits just below
    the block's last line, overlaps it horizontally and has a similar text height."""
    blocks: list[dict] = []
    for x0, y0, x1, y1, text in sorted(lines, key=lambda l: (l[1], l[0])):
        h = y1 - y0
        target = None
        for b in reversed(blocks[-6:]):
            gap = y0 - b["y1"]
            overlaps = min(x1, b["x1"]) - max(x0, b["x0"]) > 0.3 * min(x1 - x0, b["x1"] - b["x0"])
            similar = abs(h - b["h"]) <= 0.35 * max(h, b["h"])
            if -0.5 * h <= gap <= 0.9 * max(h, b["h"]) and overlaps and similar:
                target = b
                break
        if target is None:
            blocks.append({"x0": x0, "x1": x1, "y1": y1, "h": h, "lines": [text]})
        else:
            target["lines"].append(text)
            target.update(x0=min(x0, target["x0"]), x1=max(x1, target["x1"]), y1=y1, h=h)
    texts = (" ".join(b["lines"]) for b in blocks)
    # Drop icon/glyph noise ("X", "■", stray letters) that OCR picks up from graphics.
    return [t for t in texts if len(t.strip()) > 2 or any(ch.isdigit() for ch in t)]


_OCR_FIXES = [
    (re.compile("�"), "—"),  # recognizer has no em-dash glyph
    # Sans-serif "I" and "l" are identical glyphs; "Al"/"APls" are near-always "AI"/"APIs".
    (re.compile(r"\bAl\b"), "AI"),
    (re.compile(r"\bAPls\b"), "APIs"),
    (re.compile(r"\bROl\b"), "ROI"),
]


def _fix_ocr_text(text: str) -> str:
    for pattern, repl in _OCR_FIXES:
        text = pattern.sub(repl, text)
    return text


def extract_pdf(data: bytes, ocr: PdfOcr | None = None) -> list[PageText]:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupt / not a PDF
        raise ExtractionError(f"Could not open PDF: {exc}") from exc
    if doc.needs_pass:
        raise ExtractionError("PDF is password protected")

    raw_pages: list[str] = []
    ocr_flags: list[bool] = []
    try:
        for page in doc:
            text = page.get_text("text", sort=True)
            used_ocr = False
            if len(text.strip()) < MIN_TEXT_LAYER_CHARS and ocr is not None:
                text = _fix_ocr_text(ocr.page_text(page))
                used_ocr = True
            raw_pages.append(text)
            ocr_flags.append(used_ocr)
    finally:
        doc.close()

    if any(ocr_flags):
        log.info("OCR used on %d/%d pages", sum(ocr_flags), len(raw_pages))
    pages = _strip_repeated_headers_footers(raw_pages)
    result = [PageText(i + 1, _clean(t), o) for i, (t, o) in enumerate(zip(pages, ocr_flags))]
    if not any(p.text.strip() for p in result):
        raise ExtractionError("PDF contains no extractable text" + ("" if ocr else " (OCR disabled)"))
    return result


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(encoding)
            if "\x00" not in text:
                return text.replace("\r\n", "\n")
        except UnicodeDecodeError:
            continue
    raise ExtractionError("File is not valid UTF-8/UTF-16 text (binary file?)")


def _normalize_line(line: str) -> str:
    # digits -> "#" (page numbers) and punctuation dropped, so OCR variants such as
    # "zapier" / "_zapier" or "Page 3" / "Page 4" count as the same repeated line.
    return re.sub(r"[^a-z#]", "", re.sub(r"\d+", "#", line.strip().lower()))


def _strip_repeated_headers_footers(pages: list[str]) -> list[str]:
    """Drop lines/blocks (running footers, page numbers) that repeat on most pages.
    They add noise to every chunk and skew similarity toward boilerplate."""
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_normalize_line(l) for l in page.splitlines() if l.strip()})
    threshold = max(3, int(len(pages) * 0.5))
    boilerplate = {line for line, n in counts.items() if line and n >= threshold and len(line) < 120}
    return [
        "\n".join(l for l in page.splitlines() if _normalize_line(l) not in boilerplate)
        for page in pages
    ]


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    # Re-join words hyphenated across line breaks, then keep paragraph breaks only.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
