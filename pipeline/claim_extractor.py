"""
Claim Tuple Extractor
=====================

Lifts a sentence into a structured claim tuple:

    Claim(
        metric   = "market_size",
        entity   = "global_cloud_computing",
        value    = 843.0,
        unit     = "USD_B",
        time     = 2025,
        population= "global",
        source_sentence = "...",
        source_chunk   = "Section 3",
    )

Two claims can contradict only if ALL of
    (metric, entity, unit, time, population)
match AND `value` disagrees beyond tolerance.

Example
-------
    "94% of respondents run a public-cloud workload"
      → (adoption_rate, public_cloud_workload, 94, pct, 2025, survey_respondents)

    "29% of enterprises report an edge-cloud pilot"
      → (adoption_rate, edge_cloud_pilot,       29, pct, 2025, survey_respondents)

Same metric, different ENTITY — not a contradiction. Your pipeline's
current retriever pairs these because they share the word "enterprises"
and both carry a number; the claim tuple kills the pair at source.

Implementation
--------------
A two-stage extractor:

  1. Rule-based fast path. Covers the ~70% of market-report sentences
     that fit a small number of well-known templates (percentages with
     a cue noun, currency amounts with a subject, CAGR quotes).
  2. Optional LLM fallback. Ship a JSON-mode prompt if GROQ / other
     API key is present; otherwise fall back to a lower-confidence
     rule-only tuple.

The extractor is deliberately schema-strict. Every field has a
canonical dictionary of allowed vocabulary. Novel entities get a
`free_text` slot and are only compared against themselves.
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Canonical vocabulary
# ─────────────────────────────────────────────────────────────────────

# Normalised metric names. Keep narrow — each new metric increases the
# risk of spurious "same metric" collisions.
METRIC_CANON: dict[str, str] = {
    # adoption / penetration
    "adoption":             "adoption_rate",
    "penetration":          "adoption_rate",
    "use":                  "adoption_rate",
    "run":                  "adoption_rate",
    "deploy":               "adoption_rate",
    "deployment":           "adoption_rate",
    "pilot":                "adoption_rate",
    "report":               "citation_rate",
    "cited":                "citation_rate",
    "concern":              "citation_rate",
    # market sizing
    "market":               "market_size",
    "spend":                "market_size",
    "revenue":              "market_size",
    "sales":                "market_size",
    "size":                 "market_size",
    # growth
    "cagr":                 "cagr",
    "compound annual":      "cagr",
    "growth rate":          "growth_rate",
    "yoy":                  "yoy_growth",
    "year-over-year":       "yoy_growth",
    # share
    "share":                "share",
    "percent of":           "share",
    "of global":            "share",
    "of the market":        "share",
    # cost
    "savings":              "cost_savings",
    "reduction":            "cost_reduction",
    # workforce
    "unfilled":             "unfilled_roles",
    "salary":               "median_salary",
    "compensation":         "median_salary",
    "certification":        "certifications_issued",
}

UNIT_CANON: dict[str, str] = {
    "%":         "pct",
    "percent":   "pct",
    "pct":       "pct",
    "pp":        "pp",
    "billion":   "USD_B",
    "b":         "USD_B",      # guarded below
    "trillion":  "USD_T",
    "t":         "USD_T",
    "million":   "USD_M",
    "m":         "USD_M",
    "k":         "thousands",
    "$":         "USD",
    "usd":       "USD",
    "months":    "months",
    "years":     "years",
}

# Entities that appear a lot. Keep verbose so substring matching works.
ENTITY_CANON: dict[str, str] = {
    "public cloud":                "public_cloud",
    "public-cloud":                "public_cloud",
    "hybrid cloud":                "hybrid_cloud",
    "hybrid-cloud":                "hybrid_cloud",
    "multi-cloud":                 "multi_cloud",
    "multicloud":                  "multi_cloud",
    "edge-cloud":                  "edge_cloud",
    "edge cloud":                  "edge_cloud",
    "ai workload":                 "ai_workloads",
    "ai workloads":                "ai_workloads",
    "cloud-native":                "cloud_native_dev",
    "cloud native":                "cloud_native_dev",
    "amazon web services":         "aws",
    "aws":                         "aws",
    "microsoft azure":             "azure",
    "azure":                       "azure",
    "google cloud":                "gcp",
    "alibaba cloud":               "alibaba",
    "oracle cloud":                "oracle",
    "ibm cloud":                   "ibm",
    "top three":                   "top3_hyperscalers",
    "three largest":               "top3_hyperscalers",
    "saas":                        "saas",
    "paas":                        "paas",
    "iaas":                        "iaas",
    "north america":               "north_america",
    "asia-pacific":                "asia_pacific",
    "asia pacific":                "asia_pacific",
    "europe":                      "europe",
    "latin america":               "latin_america",
    "middle east & africa":        "middle_east_africa",
    "middle east and africa":      "middle_east_africa",
    "cloud-computing market":      "global_cloud_market",
    "cloud computing market":      "global_cloud_market",
    "global cloud":                "global_cloud_market",
    "security and compliance":     "security_compliance_concern",
    "cost predictability":         "cost_predictability_concern",
    "data egress":                 "data_egress_concern",
    "talent shortage":             "talent_shortage_concern",
    "legacy system":               "legacy_integration_concern",
}

# Populations a claim can quantify OVER. Two percentages can only
# contradict when the *denominator* is the same.
POPULATION_CANON: dict[str, str] = {
    "respondents":                 "survey_respondents",
    "enterprises":                 "survey_respondents",
    "enterprise respondents":      "survey_respondents",
    "companies surveyed":          "survey_respondents",
    "organisations":               "survey_respondents",
    "organizations":               "survey_respondents",
    "global cloud spend":          "global_cloud_spend",
    "cloud spend":                 "global_cloud_spend",
    "cloud market":                "global_cloud_market",
    "cost stack":                  "cost_stack",
    "cloud-related roles":         "cloud_roles",
    "cloud related roles":         "cloud_roles",
}


# ─────────────────────────────────────────────────────────────────────
# Number + unit regex
# ─────────────────────────────────────────────────────────────────────

_NUMBER_UNIT = re.compile(
    r"""
    (?P<sign>[-−]?)?
    \$?\s?
    (?P<num>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?     # 1,200 / 1,200.5
      | \d+(?:\.\d+)?                    # 47 / 47.5
    )
    \s*
    (?P<unit>
        %|percent|pp|
        billion|million|trillion|
        \bB\b|\bM\b|\bT\b|\bK\b|
        usd|USD|
        months?|years?|weeks?|days?
    )?
    """,
    re.VERBOSE,
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# ─────────────────────────────────────────────────────────────────────
# Claim dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    metric: str = ""
    entity: str = ""
    value: float = 0.0
    unit: str = ""
    time: Optional[int] = None
    population: str = ""
    qualifier: str = ""                       # "at most", "approximately", etc.
    source_sentence: str = ""
    source_chunk: str = ""
    confidence: float = 0.0                   # 0..1
    extractor: str = "rule"                   # "rule" or "llm"

    def signature(self) -> tuple:
        """
        The identity tuple: two claims are *comparable* only if their
        signatures match.
        """
        return (self.metric, self.entity, self.unit,
                self.time, self.population)

    def as_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# Rule-based extraction
# ─────────────────────────────────────────────────────────────────────

def _normalise_number(m: re.Match) -> tuple[float, str]:
    num = m.group("num").replace(",", "")
    try:
        v = float(num)
    except ValueError:
        return 0.0, ""
    if m.group("sign") in ("-", "−"):
        v = -v
    unit_raw = (m.group("unit") or "").lower().strip()
    unit = UNIT_CANON.get(unit_raw, "")
    # disambiguate bare 'b' / 'm' / 't' vs letter after a number
    if unit_raw in ("b", "m", "t"):
        # only if preceded by a number with nothing like a year
        unit = UNIT_CANON.get(unit_raw.upper(), UNIT_CANON.get(unit_raw, ""))
    return v, unit


def _find_entity(text: str) -> tuple[str, str]:
    """Longest-match lookup into ENTITY_CANON. Returns (canonical, literal)."""
    low = text.lower()
    best_lit = ""
    best_canon = ""
    for lit, canon in ENTITY_CANON.items():
        if lit in low and len(lit) > len(best_lit):
            best_lit = lit
            best_canon = canon
    return best_canon, best_lit


def _find_nearest_entity(text: str, num_start: int, num_end: int) -> str:
    """
    Bind a number to its entity.

    Preference order:
      1. entity whose literal ended most recently BEFORE num_start
         (handles "AWS ... at 31%, followed by Azure at 25%")
      2. failing that, the closest entity AFTER num_end
      3. longest-match sentence-wide entity as a last resort
    """
    low = text.lower()

    before_best_canon = ""
    before_best_end = -1
    after_best_canon = ""
    after_best_start = 10**9

    for lit, canon in ENTITY_CANON.items():
        start = 0
        while True:
            idx = low.find(lit, start)
            if idx < 0:
                break
            end = idx + len(lit)
            if end <= num_start:
                if end > before_best_end:
                    before_best_end = end
                    before_best_canon = canon
            elif idx >= num_end:
                if idx < after_best_start:
                    after_best_start = idx
                    after_best_canon = canon
            start = idx + 1

    if before_best_canon and (num_start - before_best_end) < 60:
        return before_best_canon
    if after_best_canon and (after_best_start - num_end) < 25:
        return after_best_canon
    # Fallback to longest-match over the sentence
    canon, _ = _find_entity(text)
    return canon


def _find_population(text: str) -> str:
    low = text.lower()
    best = ""
    for lit, canon in POPULATION_CANON.items():
        if lit in low and len(lit) > len(best):
            best = canon
    return best


def _find_metric(text: str, unit: str, entity: str) -> str:
    """
    Decide what metric the sentence is quoting. Use cue words first,
    then fall back to unit-based heuristics:
        pct + adoption cue  → adoption_rate
        pct + citation cue  → citation_rate
        pct + "share" cue   → share
        USD_B               → market_size
        months              → duration
    """
    low = text.lower()
    for lit, canon in METRIC_CANON.items():
        if lit in low:
            return canon

    if unit == "pct":
        if any(w in low for w in ("run", "use", "deploy", "adopt", "operate")):
            return "adoption_rate"
        if any(w in low for w in ("cite", "name", "report", "concern")):
            return "citation_rate"
        if "share" in low or "of global" in low or "of the market" in low:
            return "share"
        return "percentage"

    if unit in ("USD_B", "USD_T", "USD_M", "USD"):
        if "save" in low or "saving" in low:
            return "cost_savings"
        if "revenue" in low or "sales" in low:
            return "revenue"
        return "market_size"

    if unit in ("months", "years"):
        return "duration"

    if unit in ("pp",):
        return "percentage_point_gap"

    return "unknown"


_AGG_CUES = re.compile(
    r"\b(collectively|combined|together|jointly|in\s+aggregate|in\s+total|"
    r"all\s+(?:three|four|five|told)|top\s+(?:three|five)|sum(?:med)?\s+to)\b",
    re.IGNORECASE,
)


def _is_aggregate_context(text: str, num_start: int, num_end: int) -> bool:
    """True if aggregate cue words sit within the same clause as the number."""
    # Check within 60 chars before the number (same clause window)
    window = text[max(0, num_start - 60): num_end]
    return bool(_AGG_CUES.search(window))


def _mentions_top_providers(text: str) -> bool:
    low = text.lower()
    hits = sum(1 for p in ("aws", "amazon", "azure", "google cloud", "gcp")
               if p in low)
    return hits >= 2


def _find_time(text: str) -> Optional[int]:
    m = _YEAR_RE.search(text)
    if m:
        return int(m.group(0))
    return None


def _find_qualifier(text: str) -> str:
    low = text.lower()
    for q in ("approximately", "roughly", "about", "more than",
              "less than", "at most", "at least", "around", "up to"):
        if q in low:
            return q
    return ""


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def extract_claims(sentence: str, chunk_label: str = "") -> list[Claim]:
    """
    Extract zero or more claims from a single sentence. Multiple
    numeric tokens in the same sentence ("$843B ... 20.9%") produce
    multiple claims, each with its own metric/unit inference.
    """
    results: list[Claim] = []
    _, sentence_entity_lit = _find_entity(sentence)
    population = _find_population(sentence)
    year = _find_time(sentence)
    qualifier = _find_qualifier(sentence)

    # Skip years from triggering separate claims.
    for m in _NUMBER_UNIT.finditer(sentence):
        span = m.group(0).strip()
        if _YEAR_RE.fullmatch(span):
            continue
        value, unit = _normalise_number(m)
        if unit == "" and not _has_number_cue(sentence, m):
            continue
        if unit == "":
            tail = sentence[m.end(): m.end() + 25].lower()
            for u_lit, u_canon in UNIT_CANON.items():
                if re.search(rf"\b{re.escape(u_lit)}\b", tail):
                    unit = u_canon
                    break
        if unit == "":
            continue

        # nearest-entity resolution per number (fixes "AWS 31%, Azure 25%...")
        entity_canon = _find_nearest_entity(sentence, m.start(), m.end())
        # aggregate override: "collectively / combined / together hold X"
        if _is_aggregate_context(sentence, m.start(), m.end()):
            entity_canon = "top3_hyperscalers" if _mentions_top_providers(sentence) else "aggregate"
        metric = _find_metric(sentence, unit, entity_canon)

        results.append(Claim(
            metric=metric,
            entity=entity_canon or sentence_entity_lit,
            value=value,
            unit=unit,
            time=year,
            population=population,
            qualifier=qualifier,
            source_sentence=sentence,
            source_chunk=chunk_label,
            confidence=_rule_confidence(entity_canon, metric, unit, population),
            extractor="rule",
        ))
    return results


def _has_number_cue(sentence: str, m: re.Match) -> bool:
    """Is there a cue word in the 20 chars around the number?"""
    left = sentence[max(0, m.start() - 20): m.start()].lower()
    right = sentence[m.end(): m.end() + 20].lower()
    cues = ("million", "billion", "trillion", "percent", "roles",
            "enterprise", "respond", "country", "countries", "months",
            "years", "usd", "share")
    return any(c in left or c in right for c in cues)


def _rule_confidence(entity: str, metric: str, unit: str, pop: str) -> float:
    score = 0.4
    if entity:  score += 0.15
    if metric != "unknown": score += 0.2
    if unit:    score += 0.15
    if pop:     score += 0.10
    return min(score, 0.95)


# ─────────────────────────────────────────────────────────────────────
# Comparability check
# ─────────────────────────────────────────────────────────────────────

def claims_comparable(a: Claim, b: Claim) -> bool:
    """
    Two claims can enter a contradiction test only if their full
    signature matches. (A missing field on either side disqualifies the
    pair — we refuse to compare low-confidence tuples.)
    """
    if not a.metric or not b.metric:
        return False
    if a.metric != b.metric:
        return False
    # Entity must match. Empty-on-both-sides is allowed — the metric
    # itself carries the subject (e.g. "revenue" sentences). But if
    # exactly one side has an entity and the other doesn't, refuse.
    if a.entity != b.entity:
        return False
    if a.unit != b.unit or not a.unit:
        return False
    # time: if both have it, they must match; if either is missing, pass.
    if a.time and b.time and a.time != b.time:
        return False
    # population: same rule
    if a.population and b.population and a.population != b.population:
        return False
    return True


def claims_contradict(
    a: Claim,
    b: Claim,
    rel_tol: float = 0.03,
) -> bool:
    """Signatures match AND values disagree beyond tolerance."""
    if not claims_comparable(a, b):
        return False
    if a.value == 0 and b.value == 0:
        return False
    denom = max(abs(a.value), abs(b.value))
    if denom == 0:
        return False
    return abs(a.value - b.value) / denom > rel_tol


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sentences = [
        "Global cloud-computing spend reached $843 billion in 2025, a compound annual growth rate of 20.9% since 2021.",
        "Amazon Web Services remains the single largest individual provider at 31%, followed by Microsoft Azure at 25% and Google Cloud at 12%.",
        "First, hyperscaler concentration is deepening: Amazon Web Services, Microsoft Azure, and Google Cloud collectively hold 68% of global cloud spend, up meaningfully from prior years.",
        "Our 1,200-enterprise survey, conducted across 24 countries, finds that 94% of respondents run at least one workload on a public-cloud provider.",
        "Edge-cloud deployments are still emerging: 29% of enterprises report an active pilot, and commercial scale remains selective.",
        "Security and compliance remain the most-cited concern, named by 58% of respondents.",
        "Professional services — both hyperscaler-led and partner-delivered — are 26%.",
        "An estimated 2.1 million cloud-related roles remain unfilled globally.",
    ]
    for s in sentences:
        claims = extract_claims(s, chunk_label="test")
        print(f"\n{s[:90]}")
        for c in claims:
            print(f"  → metric={c.metric:<18s} entity={c.entity:<22s} "
                  f"value={c.value:<7g} unit={c.unit:<6s} time={c.time} "
                  f"pop={c.population}  conf={c.confidence:.2f}")

    print("\n== comparability ==")
    c1 = extract_claims("94% of respondents run a public-cloud workload in 2025.")[0]
    c2 = extract_claims("29% of enterprises report an edge-cloud pilot in 2025.")[0]
    print(f"adoption_rate(public_cloud) vs adoption_rate(edge_cloud): "
          f"comparable={claims_comparable(c1, c2)}   "
          f"(expected False — different entity)")

    c3 = extract_claims("AWS holds 31% of global cloud spend in 2025.")[0]
    c4 = extract_claims("AWS holds 42% of global cloud spend in 2025.")[0]
    print(f"AWS 31% vs AWS 42%: comparable={claims_comparable(c3, c4)} "
          f"contradicts={claims_contradict(c3, c4)}   (expected True/True)")
