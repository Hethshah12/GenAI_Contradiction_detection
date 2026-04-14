"""
ContradictAI PDF report writer.

Renders a clean, scannable contradiction report:
  Page 1 : cover + summary
  Page 2 : contents / severity index
  Page 3+: one card per contradiction, grouped High -> Medium -> Low
"""
from fpdf import FPDF
from datetime import datetime
import re


# ── theme ──────────────────────────────────────────────────────────────────
PRIMARY     = (26, 58, 143)     # deep blue
MUTED       = (110, 115, 130)
BODY        = (40, 44, 52)
SOFT_BG     = (244, 247, 253)
SOFT_RULE   = (215, 222, 236)

SEV_COLOUR = {
    "High":   (215, 48,  51),
    "Medium": (240, 136, 25),
    "Low":    (181, 151, 15),
}

SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2}


# ── helpers ────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    """Strip characters that latin-1 / core PDF fonts cannot render."""
    if text is None:
        return ""
    replacements = {
        "\u2014": "-",   "\u2013": "-",
        "\u2018": "'",   "\u2019": "'",
        "\u201c": '"',   "\u201d": '"',
        "\u2026": "...", "\u00a0": " ",
        "\u2022": "-",
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


_ANALYSIS_FIELDS = (
    "Source", "Type", "Target Section", "Conflicting Section",
    "Statement A", "Statement B", "Explanation", "Severity",
    "Confidence", "NLI Score", "LLM Analysis",
)


def _extract_field(analysis: str, field: str) -> str:
    """
    Pull a single `Field: value` block out of the analysis blob.
    Terminator is the start of the NEXT known field name, not a generic
    regex (which breaks on field names containing uppercase letters
    mid-word, e.g. "Statement B:").
    """
    if not analysis:
        return ""
    others = [f for f in _ANALYSIS_FIELDS if f != field]
    # Field headers may include a parenthetical qualifier, e.g.
    # "Statement A (Section 17): ..." - tolerate one optional parenthetical
    # between the field name and the colon.
    head = lambda f: re.escape(f) + r'(?:\s*\([^)]*\))?\s*:'
    term = "|".join(head(f) for f in others)
    pattern = rf'{head(field)}\s*(.*?)(?=\n(?:{term})|\Z)'
    m = re.search(pattern, analysis, re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip().strip('"').strip()


def _parse_contradiction(c: dict) -> dict:
    """Normalise the fields we render in the report."""
    analysis = c.get("analysis", "") or ""
    stmt_a = _extract_field(analysis, "Statement A")
    stmt_b = _extract_field(analysis, "Statement B")
    explanation = _extract_field(analysis, "Explanation")
    source = _extract_field(analysis, "Source") or "CROSS"
    conflicting = _extract_field(analysis, "Conflicting Section")

    # Statement A/B may include a leading "(Section N):" tag - drop it
    stmt_a = re.sub(r'^\([^)]+\)\s*:?\s*', '', stmt_a)
    stmt_b = re.sub(r'^\([^)]+\)\s*:?\s*', '', stmt_b)

    # For cross contradictions render both section ids, for intra just one
    target = c.get("target", "")
    if "same section" in conflicting.lower() or source.upper() == "INTRA":
        pair_display = f"{target} (intra-section)"
    else:
        related = (c.get("related_sections") or [conflicting or ""])[0]
        related = related.replace(" (same section)", "")
        pair_display = f"{target}  vs  {related}"

    return {
        "severity":    c.get("severity", "Low"),
        "confidence":  int(c.get("confidence", 0) or 0),
        "type":        c.get("type", "Direct"),
        "source":      source.upper() if source else "CROSS",
        "pair":        pair_display,
        "statement_a": stmt_a,
        "statement_b": stmt_b,
        "explanation": explanation,
    }


# ── PDF shell ──────────────────────────────────────────────────────────────
class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return  # cover page has its own heading
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*PRIMARY)
        self.cell(0, 8, "ContradictAI  .  Contradiction Report",
                  align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*SOFT_RULE)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


# ── page builders ──────────────────────────────────────────────────────────
def _cover_page(pdf: ReportPDF, doc_name: str, total_sections: int,
                contradictions: list, clean_count: int) -> None:
    pdf.add_page()
    pdf.ln(14)

    # Big title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 12, "Contradiction Detection Report",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, clean(f"Document: {doc_name}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Summary band
    band_top = pdf.get_y()
    pdf.set_fill_color(*SOFT_BG)
    pdf.set_draw_color(*SOFT_RULE)
    pdf.rect(10, band_top, 190, 34, style="FD")

    def _stat(x, label, value, value_color=PRIMARY):
        pdf.set_xy(x, band_top + 6)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.cell(40, 5, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(x, band_top + 13)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*value_color)
        pdf.cell(40, 10, str(value), new_x="LMARGIN", new_y="NEXT")

    high = sum(1 for c in contradictions if c.get("severity") == "High")
    med  = sum(1 for c in contradictions if c.get("severity") == "Medium")
    low  = sum(1 for c in contradictions if c.get("severity") == "Low")

    _stat(14,  "SECTIONS ANALYSED", total_sections)
    _stat(54,  "CONTRADICTIONS",    len(contradictions),
          value_color=SEV_COLOUR["High"] if contradictions else (67, 160, 71))
    _stat(94,  "CLEAN SECTIONS",    clean_count, value_color=(67, 160, 71))
    _stat(134, "HIGH / MED / LOW",  f"{high} / {med} / {low}")

    pdf.set_y(band_top + 40)
    pdf.ln(4)

    if not contradictions:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(67, 160, 71)
        pdf.cell(0, 10, "No contradictions detected. Document is internally consistent.",
                 new_x="LMARGIN", new_y="NEXT")


def _render_card(pdf: ReportPDF, idx: int, c: dict) -> None:
    """Render one contradiction as a compact card."""
    parsed = _parse_contradiction(c)
    sev = parsed["severity"]
    r, g, b = SEV_COLOUR.get(sev, MUTED)

    # Estimate height needed to keep card on one page if possible
    approx_height = 62 + (len(parsed["statement_a"]) + len(parsed["statement_b"])
                          + len(parsed["explanation"])) // 10
    if pdf.get_y() + approx_height > pdf.h - 20:
        pdf.add_page()

    # Severity tab
    top = pdf.get_y()
    pdf.set_fill_color(r, g, b)
    pdf.rect(10, top, 3, 40, style="F")   # left-edge colour tab

    # Header row
    pdf.set_xy(16, top)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*BODY)
    pdf.cell(0, 6, clean(f"#{idx}.  {parsed['pair']}"),
             new_x="LMARGIN", new_y="NEXT")

    # Meta chips row
    pdf.set_x(16)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(255, 255, 255)

    chips = [
        (sev.upper(), (r, g, b)),
        (parsed["type"].upper(), PRIMARY),
        (parsed["source"], (95, 105, 120)),
        (f"CONF {parsed['confidence']}%", (95, 105, 120)),
    ]
    x_cursor = 16
    for text, colour in chips:
        w = pdf.get_string_width(text) + 4
        pdf.set_fill_color(*colour)
        pdf.rect(x_cursor, pdf.get_y(), w, 4.5, style="F")
        pdf.set_xy(x_cursor + 2, pdf.get_y() + 0.2)
        pdf.cell(w - 2, 4, text)
        x_cursor += w + 2
    pdf.ln(7)

    # Statements
    def _quote(label: str, body: str):
        if not body:
            return
        pdf.set_x(16)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 4, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*BODY)
        pdf.multi_cell(184, 4.6, clean(f'"{body.strip()}"'))
        pdf.ln(0.5)

    _quote("STATEMENT A", parsed["statement_a"])
    _quote("STATEMENT B", parsed["statement_b"])

    if parsed["explanation"]:
        pdf.set_x(16)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 4, "WHY IT'S A CONTRADICTION", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*BODY)
        pdf.multi_cell(184, 4.6, clean(parsed["explanation"]))

    # Divider
    pdf.ln(3)
    pdf.set_draw_color(*SOFT_RULE)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)


# ── public entrypoint ──────────────────────────────────────────────────────
def generate_pdf_report(
    filename: str,
    doc_name: str,
    total_sections: int,
    contradictions: list,
    clean_count: int,
) -> bytes:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    _cover_page(pdf, doc_name, total_sections, contradictions, clean_count)

    if not contradictions:
        return bytes(pdf.output())

    # Sort by severity, then confidence desc
    ordered = sorted(
        contradictions,
        key=lambda c: (SEV_ORDER.get(c.get("severity", "Low"), 3),
                       -int(c.get("confidence", 0) or 0)),
    )

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 8, "Detected Contradictions",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, "Sorted by severity, then confidence.",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    for i, c in enumerate(ordered, start=1):
        _render_card(pdf, i, c)

    return bytes(pdf.output())
