"""
Negation / Antonym / Numeric-delta Gate
=======================================

Before letting a pair through as a DIRECT contradiction, require at
least one explicit conflict signal:

    * exactly-one-sided negation — one sentence has a negation token,
      the other does not, and the surrounding noun phrase overlaps.
    * antonymous predicate — rose/fell, grew/shrank, succeeded/failed,
      allowed/prohibited, etc.
    * modality flip — "must" vs "must not", "shall" vs "shall not".
    * numeric delta — both sentences make a numeric claim about the
      same metric/entity and the values disagree beyond tolerance.

Without any of these signals, DIRECT should not fire — an NLI model's
"contradiction" probability on a related-topic pair with no actual
polarity reversal is almost always noise.

Return value is a ``GateResult`` so the caller can log *why* a pair
survived (for debugging / precision tuning).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────
# Lexicons
# ─────────────────────────────────────────────────────────────────────

NEGATION_TOKENS = {
    "not", "no", "never", "none", "nothing", "neither", "nor",
    "without", "cannot", "can't", "won't", "shouldn't", "mustn't",
    "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't",
    "hasn't", "haven't", "hadn't", "unable", "fails", "fail", "failed",
    "refuses", "refused", "refuse", "prohibited", "forbidden",
    "disallowed", "excluded", "impossible",
}

# Pairs of predicates that flip polarity when swapped.
ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("rose", "fell"), ("rises", "falls"), ("rising", "falling"),
    ("grew", "shrank"), ("grows", "shrinks"), ("growing", "shrinking"),
    ("increased", "decreased"), ("increases", "decreases"),
    ("expanded", "contracted"), ("expands", "contracts"),
    ("accelerated", "decelerated"), ("accelerates", "decelerates"),
    ("improved", "worsened"), ("improves", "worsens"),
    ("succeeded", "failed"), ("succeeds", "fails"),
    ("gained", "lost"), ("gains", "loses"),
    ("positive", "negative"),
    ("allowed", "prohibited"), ("permitted", "forbidden"),
    ("required", "optional"), ("mandatory", "voluntary"),
    ("enabled", "disabled"), ("active", "inactive"),
    ("open", "closed"), ("included", "excluded"),
    ("above", "below"), ("over", "under"),
    ("before", "after"),
    ("earliest", "latest"),
    ("higher", "lower"), ("highest", "lowest"),
    ("more", "less"), ("most", "least"),
    ("fastest", "slowest"),
    ("compliant", "non-compliant"),
]

# Modal pairs that flip obligation polarity.
MODAL_POLARITY_PAIRS: list[tuple[str, str]] = [
    ("must", "must not"), ("shall", "shall not"),
    ("required to", "not required to"),
    ("permitted", "not permitted"),
    ("allowed to", "not allowed to"),
]


_WORD_RE = re.compile(r"[A-Za-z']+")
_NUM_RE  = re.compile(
    r"""
    (?:\$\s?)?                                      # optional $
    -?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s?%)?         # 1,200  1,000.5  12.5%
    | (?:\$\s?)?-?\d+(?:\.\d+)?(?:\s?%)?            # 47     47.5     12.5%
    """,
    re.VERBOSE,
)


@dataclass
class GateResult:
    passed: bool
    signals: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _tokens(s: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(s)]


def _has_negation(tokens: list[str]) -> bool:
    return any(t in NEGATION_TOKENS for t in tokens)


def _shared_content_words(ta: list[str], tb: list[str], min_share: int = 2) -> bool:
    """
    Used to confirm the two sentences are about the same subject when
    the only contradiction signal is one-sided negation.
    """
    stop = NEGATION_TOKENS | {
        "the", "a", "an", "of", "and", "or", "is", "are", "was", "were",
        "to", "in", "on", "at", "for", "by", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "should", "can", "could", "may", "might", "this", "that",
    }
    sa = {t for t in ta if t not in stop and len(t) > 2}
    sb = {t for t in tb if t not in stop and len(t) > 2}
    return len(sa & sb) >= min_share


def _find_antonym_hit(ta: list[str], tb: list[str]) -> tuple[str, str] | None:
    sa, sb = set(ta), set(tb)
    for a, b in ANTONYM_PAIRS:
        if a in sa and b in sb:
            return (a, b)
        if b in sa and a in sb:
            return (b, a)
    return None


def _find_modal_flip(sent_a: str, sent_b: str) -> tuple[str, str] | None:
    la, lb = sent_a.lower(), sent_b.lower()
    for pos, neg in MODAL_POLARITY_PAIRS:
        if pos in la and neg in lb:
            return (pos, neg)
        if neg in la and pos in lb:
            return (neg, pos)
    return None


def _as_float(tok: str) -> float | None:
    t = tok.replace("$", "").replace("%", "").replace(",", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _numeric_delta_signal(
    sent_a: str,
    sent_b: str,
    rel_tol: float = 0.03,
) -> tuple[float, float, float] | None:
    """
    If both sentences carry at least one numeric token and *some* pair
    of tokens from the two sentences is close in magnitude but disagrees
    beyond `rel_tol`, return (value_a, value_b, gap).

    This is intentionally conservative — we don't know unit/denominator
    here, that's the claim extractor's job. We're only generating a
    candidacy signal for the downstream reconciler + LLM judge.
    """
    nums_a = [_as_float(m.group(0)) for m in _NUM_RE.finditer(sent_a)]
    nums_b = [_as_float(m.group(0)) for m in _NUM_RE.finditer(sent_b)]
    nums_a = [v for v in nums_a if v is not None]
    nums_b = [v for v in nums_b if v is not None]
    if not nums_a or not nums_b:
        return None

    best: tuple[float, float, float] | None = None
    for va in nums_a:
        if va == 0:
            continue
        for vb in nums_b:
            if vb == 0:
                continue
            # compare on the same order of magnitude only
            if max(abs(va), abs(vb)) / max(1e-9, min(abs(va), abs(vb))) > 10:
                continue
            gap = abs(va - vb) / max(abs(va), abs(vb))
            if gap > rel_tol:
                if best is None or gap > best[2]:
                    best = (va, vb, gap)
    return best


# Month tokens for date-conflict detection
_MONTH_TOKEN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)


def _date_conflict_signal(sent_a: str, sent_b: str) -> tuple[str, str] | None:
    """
    If both sentences reference a different *month* and share a subject
    cue word (effective, start, begin, take effect, valid, expires,
    deadline, due), call it a date conflict.
    """
    a_months = {m.group(0).lower()[:3] for m in _MONTH_TOKEN.finditer(sent_a)}
    b_months = {m.group(0).lower()[:3] for m in _MONTH_TOKEN.finditer(sent_b)}
    if not a_months or not b_months:
        return None
    if a_months == b_months:
        return None
    cue = re.compile(
        r"\b(effective|starts?|starting|begin|begins|take\s+effect|takes\s+effect|"
        r"valid|expires?|expiry|deadline|due|active\s+from|commences?)\b",
        re.IGNORECASE,
    )
    if cue.search(sent_a) and cue.search(sent_b):
        a_pick = next(iter(a_months - b_months), next(iter(a_months)))
        b_pick = next(iter(b_months - a_months), next(iter(b_months)))
        return (a_pick, b_pick)
    return None


# ─────────────────────────────────────────────────────────────────────
# Public gate
# ─────────────────────────────────────────────────────────────────────

def gate(sent_a: str, sent_b: str) -> GateResult:
    """
    Return GateResult(passed=True) iff we find at least one concrete
    polarity / numeric / modal conflict signal between the two
    sentences. This should sit in front of the NLI scorer: if the gate
    fails, we drop the pair before spending compute on it.
    """
    ta = _tokens(sent_a)
    tb = _tokens(sent_b)

    neg_a = _has_negation(ta)
    neg_b = _has_negation(tb)

    signals: list[str] = []
    detail: dict = {}

    # 1) One-sided negation + shared subject
    if neg_a != neg_b and _shared_content_words(ta, tb, min_share=2):
        signals.append("one_sided_negation")
        detail["neg_side"] = "a" if neg_a else "b"

    # 2) Antonymous predicate pair
    anto = _find_antonym_hit(ta, tb)
    if anto is not None:
        signals.append("antonym_pair")
        detail["antonym"] = anto

    # 3) Modal polarity flip
    modal = _find_modal_flip(sent_a, sent_b)
    if modal is not None:
        signals.append("modal_flip")
        detail["modal"] = modal

    # 4) Numeric delta on overlapping magnitudes
    num = _numeric_delta_signal(sent_a, sent_b)
    if num is not None and _shared_content_words(ta, tb, min_share=1):
        signals.append("numeric_delta")
        detail["numeric"] = {"a": num[0], "b": num[1], "rel_gap": round(num[2], 4)}

    # 5) Date conflict
    dconf = _date_conflict_signal(sent_a, sent_b)
    if dconf is not None:
        signals.append("date_conflict")
        detail["date"] = dconf

    return GateResult(passed=bool(signals), signals=signals, detail=detail)


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # 16 false positives from the latest run — the gate should
        # reject all of these.
        ("Global cloud-computing spend reached $843 billion in 2025, a compound annual growth rate of 20.9% since 2021.",
         "The cloud-computing market has moved decisively from a growth story into an operating-scale story."),
        ("Figure 2 — Regional share of the 2025 cloud market.",
         "Figure 6 — Forecast regional CAGR, 2025-2030."),
        ("The SaaS share reflects accumulated migration from on-premise packaged software.",
         "IaaS maintains its share through the combination of lift-and-shift migration of legacy workloads and the GPU-heavy compute demands of model training."),
        ("Our 1,200-enterprise survey, conducted across 24 countries, finds that 94% of respondents run at least one workload on a public-cloud provider.",
         "Edge-cloud deployments, where compute is pushed closer to users or devices, are still emerging: 29% of enterprises report an active pilot, and commercial scale remains selective."),
        ("Amazon Web Services remains the single largest individual provider at 31%, followed by Microsoft Azure at 25% and Google Cloud at 12%.",
         "First, hyperscaler concentration is deepening: Amazon Web Services, Microsoft Azure, and Google Cloud collectively hold 68% of global cloud spend, up meaningfully from prior years."),

        # True contradictions — the gate MUST pass these.
        ("All employees must use VPN when working remotely.",
         "Employees working remotely do not need to use VPN."),
        ("Revenue rose to $412 million in fiscal 2024.",
         "Revenue fell to $389 million in fiscal 2024."),
        ("The policy is effective starting January 1, 2025.",
         "The policy takes effect on July 1, 2025."),
        ("Maximum daily transaction limit is $10,000.",
         "Maximum daily transaction limit is $25,000."),
    ]

    for a, b in cases:
        g = gate(a, b)
        print(f"[{'PASS' if g.passed else 'DROP'}] signals={g.signals}")
        print(f"    A: {a[:95]}")
        print(f"    B: {b[:95]}")
        print()
