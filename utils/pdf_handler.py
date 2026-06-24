"""
utils/pdf_handler.py
--------------------
Safe PDF text extraction layer for MarketMind AI.
Architecture: v2 — Smart Chunking over Hard Truncation

Enhancement
───────────
Instead of hard-truncating the full PDF text at MAX_INPUT_LENGTH (4000 chars),
the module now implements a fixed-size sliding-window chunker that preserves
all extracted text as a list of overlapping chunks. Callers can process all
chunks (e.g. for RAG ingestion) or simply use the first chunk for profile
extraction — but no text is silently discarded.

Public API
──────────
  extract_text_from_pdf(path)          → full raw text string (no truncation)
  chunk_pdf_text(text, ...)            → list of overlapping text chunks
  load_and_validate_pdf(path)          → first valid chunk (backward-compatible)
  load_and_validate_pdf_chunks(path)   → all validated chunks
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from utils.security import validate_input_text, MAX_INPUT_LENGTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE:    int = 3500   # characters per chunk (safely under 4000 limit)
CHUNK_OVERLAP: int = 200    # overlap between consecutive chunks to preserve context


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract and return all text from a PDF file WITHOUT any truncation.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        The full concatenated text of all pages.

    Raises:
        FileNotFoundError: When *file_path* does not exist.
        ValueError: When the file is not a PDF or contains no extractable text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found at path: {path.resolve()}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: '{path.suffix}'")

    reader = PdfReader(str(path))
    pages_text: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[PDF Handler] Could not extract text from page %d: %s", page_number, exc
            )

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "No extractable text found in the PDF. "
            "The file may be a scanned image without embedded text."
        )

    logger.info("[PDF Handler] Extracted %d characters from %s.", len(full_text), path.name)
    return full_text


# ---------------------------------------------------------------------------
# Sliding-window chunker
# ---------------------------------------------------------------------------

def chunk_pdf_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping fixed-size chunks using a sliding window.

    This replaces hard truncation: no content is lost; instead, long PDFs
    produce multiple chunks that can each be processed independently.

    Args:
        text:       The full extracted PDF text.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters of overlap between consecutive chunks
                    (preserves cross-boundary context).

    Returns:
        A list of non-empty text chunk strings. At minimum, one chunk is
        returned even if the text is shorter than chunk_size.

    Example:
        A 10,000-char PDF with chunk_size=3500, overlap=200 produces:
          chunk 0: chars    0 – 3499
          chunk 1: chars 3300 – 6799
          chunk 2: chars 6600 – 9999 (or end)
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start  = 0
    stride = chunk_size - overlap   # how many chars to advance each step

    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += stride

    logger.info(
        "[PDF Handler] Chunked %d chars into %d chunks "
        "(size=%d, overlap=%d).",
        len(text), len(chunks), chunk_size, overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_chunk(chunk: str) -> str | None:
    """Validate a single chunk through the security layer. Returns None if invalid."""
    # Truncate to MAX_INPUT_LENGTH only for the security check,
    # then use the original chunk if it passes
    check_text = chunk[:MAX_INPUT_LENGTH]
    if not validate_input_text(check_text):
        logger.warning("[PDF Handler] A chunk failed security validation and was dropped.")
        return None
    return chunk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_validate_pdf(file_path: str | Path) -> str | None:
    """
    Extract PDF text, chunk it, validate each chunk, and return the first
    valid chunk.

    This is the backward-compatible entry point used by app.py for initial
    profile parsing. Returns a single string suitable for the LLM profile
    extractor.

    Args:
        file_path: Path to the PDF file.

    Returns:
        The first valid text chunk, or ``None`` when extraction fails or all
        chunks fail security validation.
    """
    try:
        full_text = extract_text_from_pdf(file_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("[PDF Handler] Extraction failed: %s", exc)
        return None

    chunks = chunk_pdf_text(full_text)
    if not chunks:
        return None

    first_valid = _validate_chunk(chunks[0])
    if first_valid is None:
        logger.error("[PDF Handler] First chunk failed security validation.")
        return None

    if len(chunks) > 1:
        logger.info(
            "[PDF Handler] PDF produced %d chunks. Returning first for profile parsing. "
            "Use load_and_validate_pdf_chunks() to access all chunks.",
            len(chunks),
        )

    return first_valid


def load_and_validate_pdf_chunks(file_path: str | Path) -> list[str]:
    """
    Extract ALL valid text chunks from a PDF file.

    Use this when you need to process the full PDF content (e.g. for RAG
    ingestion, document summarisation, or when the company profile PDF is
    longer than CHUNK_SIZE characters).

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of validated text chunk strings. Empty list on failure.
    """
    try:
        full_text = extract_text_from_pdf(file_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("[PDF Handler] Extraction failed: %s", exc)
        return []

    chunks  = chunk_pdf_text(full_text)
    valid   = [c for c in [_validate_chunk(ch) for ch in chunks] if c is not None]

    logger.info(
        "[PDF Handler] load_and_validate_pdf_chunks: %d/%d chunks passed validation.",
        len(valid), len(chunks),
    )
    return valid