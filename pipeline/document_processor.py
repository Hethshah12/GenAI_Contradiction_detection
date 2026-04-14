import re
import hashlib
import logging
from collections import Counter
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Common heading patterns found in company reports
HEADING_PATTERN = re.compile(
    r'^(\d+[\.\d]*\s+[A-Z][^\n]{3,60}|'   # "1. Introduction" / "2.1 Revenue"
    r'[A-Z][A-Z\s]{4,50}|'                  # "EXECUTIVE SUMMARY"
    r'[A-Z][a-z][\w\s]{3,50}:)$',           # "Risk Factors:"
    re.MULTILINE
)

# Regexes that identify page-header / page-footer noise even when the
# exact text varies from page to page (e.g. page numbers change).
_HEADER_NOISE_PATTERNS = [
    re.compile(r'^\s*Page\s+\d+\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),                     # "3 / 14"
    re.compile(r'^\s*[-–—]\s*\d+\s*[-–—]?\s*$'),               # "- 3 -"
    re.compile(r'^\s*\d{1,3}\s*$'),                            # bare page number
]


def _normalise_line(line: str) -> str:
    """Collapse whitespace and page numbers so repeated headers match each other."""
    s = line.strip()
    s = re.sub(r'\s+', ' ', s)
    # Replace bare trailing/leading page numbers so "Report Page 3" and
    # "Report Page 14" normalise to the same key.
    s = re.sub(r'\bPage\s+\d+\b', 'Page N', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d{1,3}\s*/\s*\d{1,3}\b', 'N/N', s)
    return s


def _detect_running_lines(page_texts: list, min_pages: int = 3, ratio: float = 0.30) -> set:
    """
    Return the set of normalised lines that appear on >= `ratio` of pages
    (and at least `min_pages` pages). Those are the running headers/footers
    that leak into every chunk.
    """
    if len(page_texts) < min_pages:
        return set()

    counter: Counter = Counter()
    for page in page_texts:
        # Only look at the top 3 and bottom 3 lines of each page - that's
        # where running headers/footers live. This avoids killing real
        # repeated content like bullet labels in the body.
        lines = [ln for ln in page.splitlines() if ln.strip()]
        candidates = lines[:3] + lines[-3:]
        seen_on_this_page = set()
        for ln in candidates:
            key = _normalise_line(ln)
            if len(key) < 3 or len(key) > 120:
                continue
            if key not in seen_on_this_page:
                counter[key] += 1
                seen_on_this_page.add(key)

    threshold = max(min_pages, int(len(page_texts) * ratio))
    return {k for k, v in counter.items() if v >= threshold}


def _strip_running_lines(page_text: str, running: set) -> str:
    """Remove lines from a page that match known running header/footer patterns."""
    kept = []
    for ln in page_text.splitlines():
        stripped = ln.strip()
        if not stripped:
            kept.append(ln)
            continue
        key = _normalise_line(stripped)
        if key in running:
            continue
        if any(p.match(stripped) for p in _HEADER_NOISE_PATTERNS):
            continue
        kept.append(ln)
    return '\n'.join(kept)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF and remove page-level noise (running
    headers/footers, bare page numbers) so downstream chunking doesn't
    glue them onto body content.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = [page.get_text() for page in doc]
    doc.close()

    running = _detect_running_lines(pages_text)
    if running:
        logger.info(f"Stripping {len(running)} running header/footer line(s): "
                    + "; ".join(list(running)[:3]))

    cleaned = [_strip_running_lines(p, running) for p in pages_text]
    return "\n\n".join(cleaned)


def extract_page_images(file_bytes: bytes, min_size: int = 150) -> list[dict]:
    """
    Extract embedded images from a PDF, filtering out logos and decorative
    elements using size, aspect-ratio, and cross-page deduplication heuristics.

    Args:
        file_bytes: raw PDF content
        min_size:   minimum width/height in pixels to keep an image

    Returns:
        list of dicts: {page, image_bytes, width, height, xref}
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    seen_hashes = set()
    global_xrefs = set()        # track images that appear on many pages (logos)

    # First pass: count how many pages each xref appears on
    xref_page_count: dict[int, int] = {}
    for page in doc:
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                xref_page_count[xref] = xref_page_count.get(xref, 0) + 1
        except Exception:
            continue

    # Identify recurring images (likely headers/footers/logos)
    recurring_xrefs = {x for x, count in xref_page_count.items() if count >= 3}

    # Second pass: extract qualifying images
    for page_num, page in enumerate(doc):
        try:
            image_list = page.get_images(full=True)
        except Exception:
            continue

        for img_info in image_list:
            xref = img_info[0]

            # Skip recurring images
            if xref in recurring_xrefs:
                continue

            try:
                pix = fitz.Pixmap(doc, xref)

                # Size filter
                if pix.width < min_size or pix.height < min_size:
                    continue

                # Aspect ratio filter (skip thin dividers/rules)
                aspect = max(pix.width, pix.height) / max(1, min(pix.width, pix.height))
                if aspect > 8:
                    continue

                # Convert CMYK → RGB if needed
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_bytes = pix.tobytes("png")

                # Content-hash dedup
                h = hashlib.md5(img_bytes).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                images.append({
                    "page":        page_num,
                    "image_bytes": img_bytes,
                    "width":       pix.width,
                    "height":      pix.height,
                    "xref":        xref,
                })

            except Exception as e:
                logger.debug(f"Image extraction failed (p{page_num}, xref {xref}): {e}")
                continue

    doc.close()
    return images


def detect_heading(text: str) -> str | None:
    """
    Try to extract a heading from the first line of a text block.
    Returns the heading string or None.
    """
    first_line = text.strip().split('\n')[0].strip()
    if len(first_line) < 60 and (
        first_line.isupper() or
        re.match(r'^\d+[\.\d]*\s+\w', first_line) or
        first_line.endswith(':')
    ):
        return first_line[:60]
    return None


def heading_based_chunk(text: str, max_len: int = 1000) -> list:
    """
    Split text on heading boundaries (numbered sections like '2. Title').
    Each heading starts a new chunk. Falls back to paragraph-based chunking
    if no headings are detected.
    """
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split on numbered headings: "2. Something" or "2.1 Something"
    parts = re.split(r'\n(?=\d+[\.\d]*\s+[A-Z])', text)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # If a single heading section is too long, sub-split it
        if len(part) > max_len:
            sub = _paragraph_split(part, min_len=80, max_len=max_len)
            chunks.extend(sub)
        else:
            if len(part) >= 40:  # skip tiny fragments
                chunks.append(part)

    return chunks if len(chunks) >= 3 else []   # fallback signal


def _paragraph_split(text: str, min_len: int, max_len: int) -> list:
    """Simple paragraph-boundary splitter (no overlap)."""
    paragraphs = text.split('\n\n')
    chunks = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) <= max_len:
            buffer = (buffer + "\n" + para).strip()
        else:
            if len(buffer) >= min_len:
                chunks.append(buffer)
            buffer = para
    if len(buffer) >= min_len:
        chunks.append(buffer)
    return chunks


def semantic_chunk(text: str, min_len: int = 150, max_len: int = 1000,
                   overlap_chars: int = 200) -> list:
    """
    Split text into semantic chunks. Strategy:

    1. Try heading-based chunking first (splits on numbered sections).
       This keeps each policy section / report section as its own chunk,
       which is critical for intra-chunk contradiction detection.
    2. Fall back to paragraph-boundary chunking with overlap if no
       headings are detected.

    overlap_chars: how many characters from the end of chunk N are
    prepended to chunk N+1 so the LLM never sees a hard cut.
    """
    # Strategy 1: heading-based chunking (much better for policy docs / reports)
    heading_chunks = heading_based_chunk(text, max_len=max_len)
    if heading_chunks:
        return heading_chunks

    # Strategy 2: paragraph-boundary chunking (original behaviour)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    raw_paragraphs = text.split('\n\n')
    raw_chunks = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buffer) + len(para) <= max_len:
            buffer = (buffer + " " + para).strip()
        else:
            if len(buffer) >= min_len:
                raw_chunks.append(buffer)
            buffer = para

    if len(buffer) >= min_len:
        raw_chunks.append(buffer)

    if not raw_chunks:
        return raw_chunks

    # Add overlap: prepend the last `overlap_chars` of the previous chunk
    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        tail = raw_chunks[i - 1][-overlap_chars:].strip()
        overlapped.append(tail + " … " + raw_chunks[i])

    return overlapped


_TOC_INDICATORS = re.compile(
    r'(?:table of contents|^\s*\d+\.\s+[A-Z][^\n]{3,60}\s*\.{3,}\s*\d+\s*$)',
    re.IGNORECASE | re.MULTILINE,
)


def _is_low_info(text: str) -> bool:
    """
    Decide whether a chunk has enough propositional content to analyse.
    Cover pages, tables of contents and running-header remnants have
    almost no truth-apt sentences and produce spurious contradictions.
    """
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 200:
        return True

    # Count how many sentence-like constructs the chunk actually has.
    sentences = re.findall(r'[A-Z][^.!?]{15,}[.!?]', stripped)
    if len(sentences) < 2:
        return True

    # Table-of-contents chunks have lots of "Title ....... 5" rows
    toc_hits = len(_TOC_INDICATORS.findall(stripped))
    if toc_hits >= 3:
        return True

    # Cover-page heuristic: mostly uppercase, little punctuation
    upper_ratio = sum(1 for c in stripped if c.isupper()) / max(1, len(stripped))
    punct_count = sum(stripped.count(p) for p in '.!?;:')
    if upper_ratio > 0.35 and punct_count < 5:
        return True

    return False


def label_chunks(chunks: list, drop_low_info: bool = True) -> list:
    """
    Attach a numeric ID, a label, and a detected heading to each chunk.
    If `drop_low_info` is True (default) cover pages and TOC-like
    chunks are filtered out before numbering, so "Section 1" is never
    the report cover.
    Returns list of dicts: {id, label, display, heading, text}
    """
    labeled = []
    idx = 0
    for text in chunks:
        if drop_low_info and _is_low_info(text):
            continue
        idx += 1
        heading = detect_heading(text)
        label = f"Section {idx}"
        display = f"Section {idx}: {heading[:40]}" if heading else label

        labeled.append({
            "id": idx,
            "label": label,
            "display": display,
            "heading": heading or f"Section {idx}",
            "text": text,
        })
    return labeled