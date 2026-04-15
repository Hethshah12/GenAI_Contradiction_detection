"""
Pragmatic-role classifier.

Every candidate statement gets a *role* tag. Two statements can only
contradict each other if they share a compatible role — a narrative
framing sentence ("the market moved from growth to operating-scale")
cannot contradict a numeric claim, and a recommendation ("negotiate
reserved-capacity terms") cannot contradict a diagnosis of the problem
it is addressing.

Roles
-----
    claim          - Testable factual assertion. Contains a verifiable
                     value, entity, or relation about the subject matter.
    narrative      - Framing / summary / transition. Untestable prose.
    recommendation - Prescriptive advice. Starts with an imperative or
                     appears inside a "Recommendations" list.
    metadata       - Survey size, fielded-in date, data-source bullet.
                     Describes how the report was produced.
    caption        - Figure / Table / Exhibit reference.
    definition     - "X means Y" / "we define X as ..." sentences.
    boilerplate    - Licensing / attribution / page-chrome residue.

The classifier is rule-based first (cheap, deterministic); an optional
LLM fallback handles sentences the rules cannot label with confidence.

Rule: cross-role contradictions are forbidden, with two exceptions:
    - claim ↔ claim
    - definition ↔ definition
Everything else is dropped before NLI scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────
# Regex bank
# ─────────────────────────────────────────────────────────────────────

_NUMBER = re.compile(
    r"""
    (?:\$\s?)?                               # optional $
    -?\d{1,3}(?:,\d{3})+(?:\.\d+)?           # 1,200   1,000.5
    | (?:\$\s?)?-?\d+(?:\.\d+)?              # 47 or 47.5 or $843
    """,
    re.VERBOSE,
)

_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_CURRENCY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|trillion|B|M|T|k|K)?\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_CAGR = re.compile(r"\bC?AGR\b|compound annual growth|year-over-year|YoY", re.IGNORECASE)

_CAPTION = re.compile(
    r"^\s*(?:figure|fig\.?|table|tbl\.?|exhibit|chart|appendix)\s*\d+", re.IGNORECASE
)
_TOC_DOTS = re.compile(r"\.{3,}\s*\d+\s*$")

_RECOMMEND_STARTERS = re.compile(
    r"""^\s*
    (?:-|•|\*|\u2022|\u25cf|\(cid:127\))?      # optional bullet glyph
    \s*
    (?:we\s+recommend|recommendation|recommend|
       adopt|negotiate|invest|plan|consider|
       avoid|do\s+not|always|never|ensure|
       implement|establish|prioritise|prioritize|
       should|must)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MODAL_OBLIGATION = re.compile(
    r"""\b(
        must(?:\s+not)?|
        shall(?:\s+not)?|
        required\s+to|not\s+required\s+to|
        (?:not\s+)?permitted|(?:not\s+)?allowed\s+to|
        prohibited|forbidden|
        do\s+not\s+need\s+to|
        do\s+not\s+have\s+to|
        may\s+not|
        cannot|can't
    )\b""",
    re.VERBOSE | re.IGNORECASE,
)

_NARRATIVE_HEDGES = re.compile(
    r"""\b(
        moved\s+decisively|
        tells?\s+a\s+story|
        paints?\s+a\s+picture|
        has\s+been\s+characterised|
        reflects?\s+a\s+broader|
        represents?\s+a\s+shift|
        marks?\s+a\s+turning\s+point|
        the\s+report\s+(?:that\s+follows|argues|quantifies)|
        this\s+edition|
        in\s+summary|
        broadly\s+speaking|
        one\s+view|
        in\s+our\s+view
    )\b""",
    re.VERBOSE | re.IGNORECASE,
)

_METADATA_CUES = re.compile(
    r"""\b(
        survey\s+of|enterprise\s+respondents|
        primary\s+survey|sample\s+of|fielded\s+in|
        data\s+streams?|methodology|
        financial\s+disclosures|procurement\s+data|
        source(?:s|d|:)|
        reconciled\s+dataset|normalized\s+to|
        this\s+report\s+draws|rounding\s+conventions?|
        confidence\s+interval
    )\b""",
    re.VERBOSE | re.IGNORECASE,
)

_DEFINITION_CUES = re.compile(
    r"""\b(
        is\s+defined\s+as|
        refers?\s+to|
        means\s+the|
        we\s+define|
        for\s+the\s+purposes?\s+of\s+this\s+(?:report|section)|
        in\s+this\s+report,?\s+[a-z]+\s+means
    )\b""",
    re.VERBOSE | re.IGNORECASE,
)

_BOILERPLATE_CUES = re.compile(
    r"""\b(
        all\s+rights\s+reserved|
        subscriber\s+licen[cs]e|
        attribution\s+terms|
        data\-use\s+and\s+redistribution|
        atlas\s+intelligence\s+prepares
    )\b""",
    re.VERBOSE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

ROLE_CLAIM          = "claim"
ROLE_NARRATIVE      = "narrative"
ROLE_RECOMMENDATION = "recommendation"
ROLE_METADATA       = "metadata"
ROLE_CAPTION        = "caption"
ROLE_DEFINITION     = "definition"
ROLE_BOILERPLATE    = "boilerplate"

# Only these role pairs are allowed to contradict each other.
_ALLOWED_PAIRS = {
    (ROLE_CLAIM, ROLE_CLAIM),
    (ROLE_DEFINITION, ROLE_DEFINITION),
    # A definition can conflict with a claim that uses the term
    # differently, but this is rare and we leave it off by default.
}


@dataclass
class RoleTag:
    role: str
    score: float   # confidence 0..1
    reasons: list[str]


def classify(text: str) -> RoleTag:
    """
    Classify a single statement. Runs rules in priority order; first
    matching rule wins. Scores reflect how distinctive the signal was.
    """
    s = text.strip()
    if not s:
        return RoleTag(ROLE_BOILERPLATE, 1.0, ["empty"])

    reasons: list[str] = []

    # Captions and TOC rows are the cleanest signal — check first.
    if _CAPTION.match(s):
        return RoleTag(ROLE_CAPTION, 0.98, ["caption_prefix"])
    if _TOC_DOTS.search(s):
        return RoleTag(ROLE_CAPTION, 0.92, ["toc_dotleader"])

    # Boilerplate (license / attribution)
    if _BOILERPLATE_CUES.search(s):
        return RoleTag(ROLE_BOILERPLATE, 0.9, ["boilerplate_cue"])

    # Metadata (survey description / data sources)
    if _METADATA_CUES.search(s):
        reasons.append("metadata_cue")
        # Metadata often contains numbers (n=1200, 24 countries). Don't
        # let those bump it into 'claim'.
        return RoleTag(ROLE_METADATA, 0.85, reasons)

    # Definitions
    if _DEFINITION_CUES.search(s):
        return RoleTag(ROLE_DEFINITION, 0.9, ["definition_cue"])

    # Recommendations (list items starting with imperative verbs)
    if _RECOMMEND_STARTERS.match(s):
        return RoleTag(ROLE_RECOMMENDATION, 0.85, ["imperative_start"])

    # Modal-obligation policy sentences (must/shall/required/prohibited/...)
    # count as claims — they carry testable polarity even without numerics.
    if _MODAL_OBLIGATION.search(s):
        return RoleTag(ROLE_CLAIM, 0.8, ["modal_obligation"])

    # Narrative (hedge phrases, no numeric backbone)
    if _NARRATIVE_HEDGES.search(s):
        reasons.append("narrative_hedge")
        if not _has_numeric_backbone(s):
            return RoleTag(ROLE_NARRATIVE, 0.85, reasons)

    # Default: if it has a numeric backbone + named entity, it's a claim.
    # Otherwise, it's narrative.
    if _has_numeric_backbone(s):
        reasons.append("numeric_backbone")
        return RoleTag(ROLE_CLAIM, 0.75, reasons)

    # Fall-through: prose with no numeric anchor → narrative.
    return RoleTag(ROLE_NARRATIVE, 0.55, ["no_numeric_no_imperative"])


def _has_numeric_backbone(s: str) -> bool:
    """True if the sentence carries a testable numeric value."""
    if _PERCENT.search(s):
        return True
    if _CURRENCY.search(s):
        return True
    if _CAGR.search(s):
        return True
    # Standalone years aren't enough (narrative often cites years).
    nums = _NUMBER.findall(s)
    if len(nums) >= 2:
        return True
    # A single big number with unit context counts.
    for m in _NUMBER.finditer(s):
        tok = m.group(0)
        try:
            v = float(tok.replace("$", "").replace(",", "").strip())
        except ValueError:
            continue
        if abs(v) >= 100:
            # guard against pure year matches
            if not _YEAR.fullmatch(tok.strip()):
                return True
    return False


def compatible(role_a: str, role_b: str) -> bool:
    """Can two statements with these roles contradict each other?"""
    return (role_a, role_b) in _ALLOWED_PAIRS or (role_b, role_a) in _ALLOWED_PAIRS


# ─────────────────────────────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────────────────────────────

def annotate_statements(statements: list[str]) -> list[tuple[str, RoleTag]]:
    return [(s, classify(s)) for s in statements]


def filter_pairs_by_role(
    pairs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """
    Split a list of candidate pairs into:
        (kept_pairs, dropped_pairs_with_role_info)

    kept_pairs survive the role compatibility check; dropped_pairs
    carry the role tags so callers can log/telemetry them.
    """
    kept: list[tuple[str, str]] = []
    dropped: list[tuple[str, str, str]] = []
    for a, b in pairs:
        ra = classify(a).role
        rb = classify(b).role
        if compatible(ra, rb):
            kept.append((a, b))
        else:
            dropped.append((a, b, f"{ra} ↔ {rb}"))
    return kept, dropped


# ─────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "The cloud-computing market has moved decisively from a growth story into an operating-scale story.",
        "Global cloud-computing spend reached $843 billion in 2025, a compound annual growth rate of 20.9% since 2021.",
        "Figure 2 — Regional share of the 2025 cloud market.",
        "- Negotiate committed-spend and reserved-capacity terms in parallel rather than sequentially.",
        "Primary survey: 1,200 enterprise respondents across 24 countries, fielded in Q1 2026.",
        "For the purposes of this report, SaaS is defined as application software delivered over the public internet.",
        "Atlas Intelligence prepares this report for institutional subscribers.",
        "Amazon Web Services remains the single largest individual provider at 31%.",
        "Security and compliance remain the most-cited concern, named by 58% of respondents.",
    ]
    for t in tests:
        tag = classify(t)
        print(f"[{tag.role:15s}] score={tag.score:.2f}  reasons={tag.reasons}")
        print(f"    {t[:90]}")
        print()
