"""
PolicyLens Document Ingestion Module
Handles robust PDF parsing, header/footer deduplication, section detection,
and structure-aware recursive chunking with overlap.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import (
    DATA_DIR,
    TARGET_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    APPROX_WORDS_PER_TOKEN,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents an ingested, structured text chunk with rich provenance metadata."""
    chunk_id: str
    doc_name: str
    doc_type: str  # 'regulation' or 'policy'
    page_start: int
    page_end: int
    section: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


# Section header detection regex patterns
SECTION_HEADER_PATTERNS = [
    re.compile(r"^(SECTION\s+\d+[:\-\s].*?)(?:$|\n)", re.IGNORECASE),
    re.compile(r"^(CLAUSE\s+[0-9A-Z\.]+[:\-\s].*?)(?:$|\n)", re.IGNORECASE),
    re.compile(r"^([0-9]+\.[0-9]+(?:\.[0-9]+)?\s+[A-Za-z\s\(\)]+[:\-])", re.IGNORECASE),
    re.compile(r"^(STANDARD TERMS AND CONDITIONS|WAITING PERIODS?|GENERAL EXCLUSIONS?|GRIEVANCE REDRESSAL MECHANISM|CASHLESS FACILITY|PORTABILITY|MORATORIUM PERIOD|DEFINITIONS AND KEY TERMINOLOGY|COVERAGE AND BENEFITS)", re.IGNORECASE),
]


def extract_section_title(line: str) -> Optional[str]:
    """Detect if a text line represents a major section heading and return the clean title."""
    stripped = line.strip()
    if not stripped:
        return None

    for pat in SECTION_HEADER_PATTERNS:
        match = pat.match(stripped)
        if match:
            raw_title = match.group(1).strip().rstrip(":-")
            return raw_title[:80].strip()

    if stripped.isupper() and 4 <= len(stripped) <= 60 and not stripped.endswith("."):
        return stripped

    return None


def extract_pages_pymupdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract raw page text using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append((page_num + 1, text))
    doc.close()
    return pages


def extract_pages_pdfplumber(pdf_path: Path) -> List[Tuple[int, str]]:
    """Fallback page extraction using pdfplumber."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((idx + 1, text))
    return pages


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract page text with primary pymupdf and fallback to pdfplumber."""
    try:
        pages = extract_pages_pymupdf(pdf_path)
        if any(len(text.strip()) > 0 for _, text in pages):
            return pages
    except Exception as e:
        logger.warning("PyMuPDF extraction failed for %s (%s). Falling back to pdfplumber.", pdf_path.name, e)

    try:
        return extract_pages_pdfplumber(pdf_path)
    except Exception as e:
        logger.error("pdfplumber extraction also failed for %s: %s", pdf_path.name, e)
        return []


def clean_page_text(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    Remove recurring headers and footers (appearing on >50% of pages),
    fix line-break hyphenations, and normalize whitespace.
    """
    if not pages:
        return []

    num_pages = len(pages)
    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}

    for _, text in pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        for top_line in lines[:2]:
            if len(top_line) > 3 and not top_line.isdigit():
                header_counts[top_line] = header_counts.get(top_line, 0) + 1
        for btm_line in lines[-2:]:
            if len(btm_line) > 3 and not btm_line.isdigit():
                footer_counts[btm_line] = footer_counts.get(btm_line, 0) + 1

    recurring_headers = {line for line, count in header_counts.items() if count > max(3, num_pages * 0.6)}
    recurring_footers = {line for line, count in footer_counts.items() if count > max(3, num_pages * 0.6)}

    cleaned_pages = []
    for page_num, text in pages:
        lines = text.splitlines()
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped in recurring_headers or stripped in recurring_footers:
                continue
            if re.match(r"^(?:Page\s+\d+(?:\s+of\s+\d+)?|\-?\s*\d+\s*\-?)$", stripped, re.IGNORECASE):
                continue
            filtered_lines.append(line)

        raw_joined = "\n".join(filtered_lines)
        dehyphenated = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", r"\1\2", raw_joined)
        normalized = re.sub(r"[ \t]+", " ", dehyphenated)
        cleaned_pages.append((page_num, normalized.strip()))

    return cleaned_pages


def chunk_document(
    doc_name: str,
    doc_type: str,
    cleaned_pages: List[Tuple[int, str]],
    target_words: int = int(TARGET_CHUNK_TOKENS * APPROX_WORDS_PER_TOKEN),
    overlap_words: int = int(CHUNK_OVERLAP_TOKENS * APPROX_WORDS_PER_TOKEN),
) -> List[DocumentChunk]:
    """
    Structure-aware recursive chunker:
    - Splits along section headings without dropping headings.
    - Preserves semantic clauses and legal caveats.
    - Applies sliding-window overlap across sentence boundaries.
    """
    chunks: List[DocumentChunk] = []
    chunk_index = 0

    # Collect section blocks
    current_title = "Preamble & Title"
    pending_text = []
    pending_pages = []

    for page_num, page_text in cleaned_pages:
        # Split on section heading lines
        raw_sections = re.split(r"(?:\n(?=SECTION\s+\d+|CLAUSE\s+[0-9A-Z\.]+))", page_text)
        for sec in raw_sections:
            sec_str = sec.strip()
            if not sec_str:
                continue
            lines = sec_str.splitlines()
            first_line = lines[0].strip()
            detected_sec = extract_section_title(first_line)

            # If this is a new section and we already have pending text, emit a chunk
            if detected_sec and detected_sec != current_title:
                if pending_text:
                    full_text = " ".join(" ".join(pending_text).split())
                    words = full_text.split()
                    if len(words) > 15:
                        chunk_index += 1
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{Path(doc_name).stem}_{chunk_index:04d}",
                                doc_name=doc_name,
                                doc_type=doc_type,
                                page_start=min(pending_pages) if pending_pages else page_num,
                                page_end=max(pending_pages) if pending_pages else page_num,
                                section=current_title,
                                text=full_text,
                            )
                        )
                    # Check if the section was small (< 40 words) and keep prefix, or reset
                    pending_text = []
                    pending_pages = []
                current_title = detected_sec

            pending_text.append(sec_str)
            pending_pages.append(page_num)

            # If accumulated section exceeds target_words, split with overlap
            words = " ".join(" ".join(pending_text).split()).split()
            while len(words) >= target_words:
                chunk_index += 1
                chunk_words = words[:target_words]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{Path(doc_name).stem}_{chunk_index:04d}",
                        doc_name=doc_name,
                        doc_type=doc_type,
                        page_start=min(pending_pages),
                        page_end=max(pending_pages),
                        section=current_title,
                        text=" ".join(chunk_words),
                    )
                )
                words = words[target_words - overlap_words:]
                pending_text = [" ".join(words)]
                pending_pages = [page_num]

    # Flush any remaining text
    if pending_text:
        full_text = " ".join(" ".join(pending_text).split()).strip()
        if len(full_text.split()) > 15:
            chunk_index += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{Path(doc_name).stem}_{chunk_index:04d}",
                    doc_name=doc_name,
                    doc_type=doc_type,
                    page_start=min(pending_pages) if pending_pages else 1,
                    page_end=max(pending_pages) if pending_pages else 1,
                    section=current_title,
                    text=full_text,
                )
            )

    return chunks


def infer_doc_type(doc_name: str, sample_text: str = "") -> str:
    """Classify document into 'regulation' or 'policy'."""
    lower_name = doc_name.lower()
    lower_text = sample_text.lower()
    if any(k in lower_name or k in lower_text for k in ["irdai", "regulation", "circular", "guideline", "ombudsman"]):
        return "regulation"
    return "policy"


def ingest_all_documents(data_dir: Path = DATA_DIR) -> List[DocumentChunk]:
    """
    Ingest all PDF documents in the specified directory.
    If no PDFs exist, generate sample realistic insurance documents.
    """
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("\n" + "=" * 70)
        print("[NOTICE] data/ directory is empty!")
        print("[NOTICE] Generating 3 synthetic Indian Health Insurance & IRDAI PDFs")
        print("[NOTICE] Replace these with your real policy PDFs and click 'Rebuild index'!")
        print("=" * 70 + "\n")
        from src.sample_docs import generate_sample_documents
        generate_sample_documents(data_dir)
        pdf_files = list(data_dir.glob("*.pdf"))

    all_chunks: List[DocumentChunk] = []
    total_pages = 0

    print(f"\nIngesting {len(pdf_files)} PDF document(s) from {data_dir}...")
    for pdf_path in sorted(pdf_files):
        pages = extract_pages(pdf_path)
        if not pages:
            logger.warning("No extractable text found in %s", pdf_path.name)
            continue

        cleaned = clean_page_text(pages)
        total_pages += len(cleaned)
        sample_snippet = " ".join([p[1] for p in cleaned[:2]])
        doc_type = infer_doc_type(pdf_path.name, sample_snippet)

        doc_chunks = chunk_document(
            doc_name=pdf_path.name,
            doc_type=doc_type,
            cleaned_pages=cleaned,
        )
        all_chunks.extend(doc_chunks)
        print(f"  * {pdf_path.name}: {len(cleaned)} pages -> {len(doc_chunks)} chunks (Type: {doc_type})")

    avg_words = sum(len(c.text.split()) for c in all_chunks) / max(1, len(all_chunks))
    avg_tokens = avg_words / APPROX_WORDS_PER_TOKEN

    print("\nIngestion Summary:")
    print(f"  - Total Documents: {len(pdf_files)}")
    print(f"  - Total Pages:     {total_pages}")
    print(f"  - Total Chunks:    {len(all_chunks)}")
    print(f"  - Avg Chunk Size:  {avg_tokens:.1f} tokens (~{avg_words:.1f} words)")
    print("-" * 50)

    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = ingest_all_documents()
    print(f"Successfully processed {len(chunks)} chunks.")
