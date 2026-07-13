"""
nlp/fulltext/pdf_parser.py
----------------------------
Downloads and parses PDF full text from a URL using pymupdf4llm (layout-aware
extraction built on PyMuPDF/fitz), with fitz used directly for metadata only.

IMPROVEMENTS OVER THE ORIGINAL:
  1. Windows-safe tempfile — uses delete=False + explicit cleanup in finally.
     The original NamedTemporaryFile was open when fitz tried to read it,
     causing PermissionError on Windows.

  2. Layout-aware reading order via pymupdf4llm.to_markdown(), replacing the
     old y-position block-sort heuristic. The old approach ("sort blocks by
     page, then y//10, then x") worked for simple 2-column layouts but broke
     down on 3-column tables, sidebar figures, and irregular grids — it had
     no real understanding of layout, just a positional guess. pymupdf4llm
     uses MuPDF's own layout analysis, verified directly against a real
     2-column journal PDF to produce clean reading order AND correctly
     reconstruct a results table as structured markdown (rows/columns
     intact) instead of flattening it into unstructured running text.

  3. All resource handles closed explicitly (doc.close(), os.unlink).

  4. Content-Type check before parsing — HTML error pages no longer crash fitz.

  5. HTTP error handling — 403 (paywalled), 429 (rate-limited), redirects all
     logged and returned as None cleanly.

  6. Minimum text length check — scanned-image PDFs (no text layer, and no
     OCR engine configured) return None instead of an empty success result
     that silently corrupts downstream NER. Verified: pymupdf4llm degrades
     gracefully (returns near-empty markdown) rather than raising when a
     page has no extractable text and no OCR is available.

  7. PDF metadata extraction — title, author, keywords from PDF properties
     added to result dict at zero extra cost.

  8. Section-aware text extraction — first tries structural markdown headers
     (# / ## lines that pymupdf4llm detected from actual font/layout cues,
     not word-guessing), falling back to the original regex-over-plain-text
     approach if no markdown headers were found. This means "Experimental
     Procedures" or other non-standard header wording can still be picked
     up structurally even though it wouldn't match the regex keyword list.

  9. OCR as an explicit last-resort tier, not an automatic default.
     pymupdf4llm ships an internal auto-OCR heuristic (use_ocr=
     SELECT_KEEP_OLD by default) that decides per-page whether OCR would
     "help" — this was tested directly and found to MISFIRE on a normal,
     fully-text-layered PDF (a 32,890-char extraction silently dropped to
     21,054 chars because its heuristic decided two pages "needed" OCR
     when they didn't, and OCR is lossier than the real text layer).
     Because of this, the primary extraction pass explicitly disables
     auto-OCR (OCRMode.NEVER). OCR is only invoked as a SECOND, explicit
     pass — and only when the primary pass returns near-zero text (i.e.
     the page genuinely looks scanned) — using OCRMode.FORCE_DROP_OLD.
     This uses PyMuPDF's own integrated Tesseract bridge (no separate
     pytesseract dependency); if Tesseract isn't installed on the host,
     it degrades gracefully to empty text rather than raising, verified
     directly by temporarily hiding tesseract from PATH.
"""

import os
import re
import tempfile
from typing import Optional
from loguru import logger

import requests
import pymupdf4llm
from pymupdf4llm.ocr import OCRMode
from nlp.fulltext.domain_throttle import throttle as domain_throttle

# Suppress harmless MuPDF warnings (e.g. "cmsOpenProfileFromMem failed" for
# PDFs with malformed ICC color profiles). These are cosmetic — text extraction
# still succeeds — but they clutter terminal output during presentations.
import fitz  # PyMuPDF
fitz.TOOLS.mupdf_display_errors(False)

# Minimum characters of extracted text to consider a fetch successful.
# Scanned-image PDFs and empty files produce 0–100 chars.
MIN_TEXT_LENGTH = 300

# Below this length, the primary (no-OCR) extraction is treated as having
# failed and an OCR fallback pass is attempted. Deliberately lower than
# MIN_TEXT_LENGTH — a handful of stray characters (page numbers, running
# headers picked up as text) shouldn't block the OCR fallback from running.
OCR_TRIGGER_THRESHOLD = 50


class PDFParser:
    """
    Downloads and parses PDF full text with proper multi-column support
    and safe resource management.
    """

    def fetch(self, url: str) -> Optional[dict]:
        """
        Downloads the PDF at `url` and extracts its text content.

        Returns:
            dict with full_text, optional section keys, metadata, and
            fetch_source/fetch_status — or None on any failure.
        """
        if not url or not url.strip():
            return None

        # ── Download ──────────────────────────────────────────────────────────
        domain_throttle(url)  # per-domain rate limit — blocks if too soon
        try:
            resp = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "MicrobiomeMiner/1.0 (Academic research)"},
                allow_redirects=True,
            )
        except requests.exceptions.Timeout:
            logger.debug(f"[pdf_parser] Timeout fetching {url[:80]}")
            return None
        except Exception as e:
            logger.debug(f"[pdf_parser] Download error for {url[:80]}: {e}")
            return None

        if resp.status_code == 429:
            logger.warning("[pdf_parser] Rate-limited (429) — will retry later")
            return None
        if resp.status_code == 403:
            logger.debug(f"[pdf_parser] Paywalled (403): {url[:80]}")
            return None
        if resp.status_code != 200:
            logger.debug(f"[pdf_parser] HTTP {resp.status_code} for {url[:80]}")
            return None

        # ── Content-type guard ────────────────────────────────────────────────
        content_type = resp.headers.get("Content-Type", "").lower()
        if "html" in content_type and "pdf" not in content_type:
            logger.debug(f"[pdf_parser] Not a PDF (Content-Type: {content_type}): {url[:80]}")
            return None
        if not resp.content:
            return None

        # ── Safe tempfile (Windows-compatible) ────────────────────────────────
        # delete=False + manual unlink in finally — the original code kept the
        # file open while fitz tried to read it, crashing on Windows.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name   # file is now closed and readable

            return self._parse_pdf(tmp_path, url)

        except Exception as e:
            logger.warning(f"[pdf_parser] Parse failed for {url[:80]}: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ── PDF text extraction ───────────────────────────────────────────────────

    def _parse_pdf(self, path: str, url: str) -> Optional[dict]:
        """
        Parses a local PDF file into structured text.

        Text extraction uses pymupdf4llm.to_markdown(), which applies MuPDF's
        own layout analysis instead of a positional sort heuristic. This
        handles multi-column journal layouts correctly and preserves table
        structure (rows/columns) as markdown tables instead of flattening
        them into unstructured running text. fitz is still used directly,
        but only for cheap metadata extraction (title/keywords).
        """
        try:
            import fitz  # PyMuPDF — used here only for metadata
        except ImportError:
            logger.warning("[pdf_parser] PyMuPDF not installed — cannot parse PDFs")
            return None

        # ── Metadata (cheap, separate from text extraction) ────────────────────
        pdf_title = ""
        pdf_keywords = ""
        try:
            doc = fitz.open(path)
            meta = doc.metadata or {}
            pdf_title    = (meta.get("title")    or "").strip()
            pdf_keywords = (meta.get("keywords") or "").strip()
            page_count   = len(doc)
            doc.close()
        except Exception as e:
            logger.debug(f"[pdf_parser] Metadata extraction failed: {e}")
            page_count = 0

        # ── Pass 1: Layout-aware text extraction, OCR explicitly disabled ──────
        # OCRMode.NEVER is required here — the library's default heuristic
        # was verified to misfire on normal PDFs (see module docstring #9).
        used_ocr = False
        try:
            full_text = pymupdf4llm.to_markdown(path, use_ocr=OCRMode.NEVER)
        except Exception as e:
            logger.debug(f"[pdf_parser] pymupdf4llm.to_markdown failed: {e}")
            full_text = ""

        # ── Pass 2 (fallback): force OCR only if pass 1 found ~nothing ─────────
        # Only triggered for pages that look genuinely scanned (no usable
        # text layer at all). Verified: adds ~1s/page, and degrades to an
        # empty result (not an exception) if Tesseract isn't installed.
        if len(full_text.strip()) < OCR_TRIGGER_THRESHOLD:
            try:
                ocr_text = pymupdf4llm.to_markdown(path, use_ocr=OCRMode.FORCE_DROP_OLD)
            except Exception as e:
                logger.debug(f"[pdf_parser] OCR fallback pass failed: {e}")
                ocr_text = ""

            if len(ocr_text.strip()) > len(full_text.strip()):
                full_text = ocr_text
                used_ocr = True
                logger.debug(
                    f"[pdf_parser] Primary extraction found ~no text — "
                    f"OCR fallback recovered {len(ocr_text)} chars"
                )

        if not full_text or len(full_text.strip()) < MIN_TEXT_LENGTH:
            logger.debug(
                f"[pdf_parser] Extracted text too short "
                f"({len(full_text or '')} chars) even after OCR fallback — "
                f"likely scanned image with no usable content, or OCR "
                f"unavailable (Tesseract not installed)"
            )
            return None

        # ── Attempt section identification ────────────────────────────────────
        sections = self._extract_sections(full_text)

        result: dict = {
            "full_text":    full_text,
            "fetch_source": "pdf",
            "fetch_status": "success",
            "source_url":   url,
        }
        if pdf_title:
            result["pdf_title"] = pdf_title
        if pdf_keywords:
            result["pdf_keywords"] = pdf_keywords
        if used_ocr:
            result["extracted_via_ocr"] = True

        result.update(sections)

        logger.debug(
            f"[pdf_parser] Extracted {len(full_text)} chars "
            f"from {page_count} pages | sections: {list(sections.keys())} | "
            f"ocr={used_ocr}"
        )
        return result

    # ── Section identification ────────────────────────────────────────────────

    # Structural markdown headers pymupdf4llm emits, e.g. "## METHODS" or
    # "# 2. Materials and Methods". Matched first — this doesn't depend on
    # guessing the exact wording, only that the layout engine identified it
    # as a heading (via font size/weight/position), so non-standard phrasing
    # like "Experimental Procedures" is still caught structurally as long as
    # its keywords loosely match one of the categories below.
    _MD_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)

    # Loose keyword matching against a header's text (not full-line anchored
    # like the old plain-text regex), classifying it into one of our known
    # section categories. Order matters: more specific checks first.
    _SECTION_KEYWORDS = {
        "abstract":   ("abstract", "summary"),
        "methods":    ("method", "material", "procedure", "study design", "experimental"),
        "results":    ("result", "finding", "outcome"),
        "discussion": ("discussion", "conclusion", "interpretation", "significance"),
    }

    # Fallback plain-text patterns — used only when pymupdf4llm produced no
    # structural markdown headers at all (rare; e.g. very simple single-column
    # PDFs where the layout engine didn't detect distinct heading formatting).
    _SECTION_MAP = {
        "methods":    re.compile(
            r"^\s*(?:\d+\.?\s+)?(?:methods?|materials?\s+and\s+methods?|"
            r"patients?\s+and\s+methods?|study\s+design)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "results":    re.compile(
            r"^\s*(?:\d+\.?\s+)?(?:results?|findings?|outcomes?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "discussion": re.compile(
            r"^\s*(?:\d+\.?\s+)?(?:discussion|conclusions?|interpretation)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "abstract":   re.compile(
            r"^\s*(?:abstract|summary)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    }

    def _classify_header(self, header_text: str) -> Optional[str]:
        """Maps a heading's text to one of our known section categories."""
        lowered = header_text.strip().lower()
        for section_name, keywords in self._SECTION_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return section_name
        return None

    def _extract_sections_from_markdown(self, text: str) -> dict:
        """
        Splits markdown text into sections using structural headers
        (# / ## / etc. lines pymupdf4llm identified from actual layout,
        not word-guessing). Each header is classified via loose keyword
        matching, so wording variance is tolerated as long as it contains
        a recognizable keyword.

        Handles nested subsections: a classified header (e.g. "MATERIALS
        AND METHODS") is often immediately followed by unclassified
        subheaders (e.g. "Data Extraction", "Analysis") that are still
        part of that same section. Content is accumulated under the most
        recent classified header until the NEXT classified header appears,
        rather than stopping at the very next header of any kind — this
        was the bug in the first version, which cut "methods" content
        down to 2 characters because "MATERIALS AND METHODS" was
        immediately followed by an unclassified subsection header.
        """
        matches = list(self._MD_HEADER_RE.finditer(text))
        if not matches:
            return {}

        # First pass: classify every header, keep its position.
        classified = [
            (m.start(), m.end(), self._classify_header(m.group(1)))
            for m in matches
        ]

        sections: dict = {}
        for i, (start, end, section_name) in enumerate(classified):
            if not section_name:
                continue
            # Find the next CLASSIFIED header (skip over unclassified
            # subsection headers in between — they belong to this section).
            next_start = len(text)
            for j in range(i + 1, len(classified)):
                if classified[j][2]:  # next classified header found
                    next_start = classified[j][0]
                    break

            content = text[end:next_start].strip()
            if content and len(content) > 50:
                # If the same section name appears twice (rare), keep the
                # longer content rather than overwriting with a shorter one.
                if section_name not in sections or len(content) > len(sections[section_name]):
                    sections[section_name] = content

        return sections

    def _extract_sections(self, text: str) -> dict:
        """
        Attempts to split extracted PDF text into labeled sections.
        Tries structural markdown headers first (handles non-standard
        wording better since it only needs a loose keyword match against
        an already-identified heading, not a full-line regex guess).
        Falls back to the original full-line plain-text regex approach if
        no markdown headers were found at all.

        Returns a dict with keys: abstract, methods, results, discussion.
        All keys are optional — only included if the section was detected.
        """
        sections = self._extract_sections_from_markdown(text)
        if sections:
            return sections

        # ── Fallback: original plain-text, full-line-anchored regex ───────────
        boundaries = []
        for section_name, pattern in self._SECTION_MAP.items():
            for match in pattern.finditer(text):
                boundaries.append((match.start(), match.end(), section_name))

        if not boundaries:
            return {}

        boundaries.sort(key=lambda x: x[0])
        sections = {}

        for i, (start, end, name) in enumerate(boundaries):
            next_start = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            content    = text[end:next_start].strip()
            if content and len(content) > 50:   # skip empty/tiny sections
                sections[name] = content

        return sections
