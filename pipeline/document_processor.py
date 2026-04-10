import re
import hashlib
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Common heading patterns found in company reports
HEADING_PATTERN = re.compile(
    r'^(\d+[\.\d]*\s+[A-Z][^\n]{3,60}|'   # "1. Introduction" / "2.1 Revenue"
    r'[A-Z][A-Z\s]{4,50}|'                  # "EXECUTIVE SUMMARY"
    r'[A-Z][a-z][\w\s]{3,50}:)$',           # "Risk Factors:"
    re.MULTILINE
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return "\n\n".join(pages_text)


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


def label_chunks(chunks: list) -> list:
    """
    Attach a numeric ID, a label, and a detected heading to each chunk.
    Returns list of dicts: {id, label, heading, text}
    """
    labeled = []
    for i, text in enumerate(chunks):
        heading = detect_heading(text)
        label = f"Section {i + 1}"
        if heading:
            display = f"Section {i + 1}: {heading[:40]}"
        else:
            display = label

        labeled.append({
            "id": i + 1,
            "label": label,
            "display": display,
            "heading": heading or f"Section {i + 1}",
            "text": text
        })
    return labeled