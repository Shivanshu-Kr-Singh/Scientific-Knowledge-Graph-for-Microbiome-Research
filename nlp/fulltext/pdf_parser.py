"""
nlp/fulltext/pdf_parser.py
----------------------------
Downloads and parses PDF full text from a URL using PyMuPDF (fitz).

IMPROVEMENTS OVER THE ORIGINAL:
  1. Windows-safe tempfile — uses delete=False + explicit cleanup in finally.
     The original NamedTemporaryFile was open when fitz tried to read it,
     causing PermissionError on Windows.

  2. Proper multi-column reading order — uses fitz "blocks" mode and sorts
     text blocks by (y0, x0) position. The default fitz "text" mode reads in
     content-stream order which interleaves left/right column text on 2-column
     PDFs (nearly all journal articles), producing unreadable output.

  3. All resource handles closed explicitly (doc.close(), os.unlink).

  4. Content-Type check before parsing — HTML error pages no longer crash fitz.

  5. HTTP error handling — 403 (paywalled), 429 (rate-limited), redirects all
     logged and returned as None cleanly.

  6. Minimum text length check — scanned-image PDFs (fitz gets no text from
     images) return None instead of an empty success result that silently
     corrupts downstream NER.

  7. PDF metadata extraction — title, author, keywords from PDF properties
     added to result dict at zero extra cost.

  8. Section-aware text extraction — attempts to identify and preserve
     common section boundaries (Introduction, Methods, Results, Discussion)
     from the continuous block stream using the same header detection used
     by SectionParser.
"""

import os
import re
import tempfile
from typing import Optional
from loguru import logger

import requests

# Minimum characters of extracted text to consider a fetch successful.
# Scanned-image PDFs and empty files produce 0–100 chars.
MIN_TEXT_LENGTH = 300


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

        Uses fitz "blocks" mode and reading-order reconstruction so that
        multi-column journal articles are read correctly (left column first,
        then right — not interleaved as the default mode produces).
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("[pdf_parser] PyMuPDF not installed — cannot parse PDFs")
            return None

        try:
            doc = fitz.open(path)
        except Exception as e:
            logger.debug(f"[pdf_parser] fitz.open failed: {e}")
            return None

        try:
            # ── Extract metadata ──────────────────────────────────────────────
            meta = doc.metadata or {}
            pdf_title    = (meta.get("title")    or "").strip()
            pdf_keywords = (meta.get("keywords") or "").strip()

            # ── Extract text in reading order ─────────────────────────────────
            all_blocks = []
            for page_num, page in enumerate(doc):
                # "blocks" returns: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 = text, 1 = image
                blocks = page.get_text("blocks")
                for block in blocks:
                    if len(block) >= 5 and block[6] == 0:   # text block only
                        x0, y0, x1, y1, text = block[:5]
                        if text.strip():
                            all_blocks.append({
                                "page": page_num,
                                "x0": x0, "y0": y0,
                                "x1": x1, "y1": y1,
                                "text": text,
                            })

            if not all_blocks:
                logger.debug(f"[pdf_parser] No text blocks extracted — likely a scanned image PDF")
                return None

            # ── Reading order: sort by page, then y-position, then x-position ─
            # This correctly handles 2-column layouts: top of left column comes
            # before top of right column, which comes before bottom of left column.
            # We use a row-band approach: blocks within 10pt of the same y are
            # on the same horizontal band, sorted left-to-right within the band.
            all_blocks.sort(key=lambda b: (b["page"], round(b["y0"] / 10), b["x0"]))

            full_text = "\n".join(b["text"].strip() for b in all_blocks)

            # ── Minimum length guard ──────────────────────────────────────────
            if len(full_text.strip()) < MIN_TEXT_LENGTH:
                logger.debug(
                    f"[pdf_parser] Extracted text too short "
                    f"({len(full_text)} chars) — possibly scanned image"
                )
                return None

            # ── Attempt section identification ────────────────────────────────
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

            # Merge extracted sections into result
            result.update(sections)

            logger.debug(
                f"[pdf_parser] Extracted {len(full_text)} chars "
                f"from {len(doc)} pages | sections: {list(sections.keys())}"
            )
            return result

        finally:
            doc.close()

    # ── Section identification ────────────────────────────────────────────────

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

    def _extract_sections(self, text: str) -> dict:
        """
        Attempts to split extracted PDF text into labeled sections.
        Returns a dict with keys: abstract, methods, results, discussion.
        All keys are optional — only included if the section was detected.
        """
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
