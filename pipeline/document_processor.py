import re
import fitz  # PyMuPDF


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


def semantic_chunk(text: str, min_len: int = 150, max_len: int = 1000) -> list:
    """
    Split text into semantic chunks using paragraph boundaries.
    Returns a list of text strings.
    """
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    raw_paragraphs = text.split('\n\n')
    chunks = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buffer) + len(para) <= max_len:
            buffer = (buffer + " " + para).strip()
        else:
            if len(buffer) >= min_len:
                chunks.append(buffer)
            buffer = para

    if len(buffer) >= min_len:
        chunks.append(buffer)

    return chunks


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