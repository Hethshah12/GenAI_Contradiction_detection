"""
Deterministic numeric-contradiction extractor.

Purpose
-------
Complements the NLI + LLM stack with a regex-based pass that finds
(entity, metric, value) tuples and flags the buckets where the same
entity/metric has disagreeing numbers across the document.

Characteristics:
  * No LLM calls — pure Python, ~100ms on a 50-page report.
  * 100% deterministic, stable severity (gap % between min/max values).
  * Built to *add to* the contradiction list, never to replace.
  * Fails open: any exception during extraction is swallowed and the
    main pipeline continues unaffected.
  * Output schema is identical to `run_full_pipeline`'s contradictions,
    so the downstream report generator, Neo4j store and UI all work
    with zero changes.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VALUE PARSING
# ─────────────────────────────────────────────────────────────────────────────
# We normalise every numeric mention to (magnitude_as_float, unit_bucket).
# Two values are compared only when they share the same unit_bucket.

_UNIT_SUFFIX = {
    "trillion": 1e12, "trn": 1e12, "t": 1e12,
    "billion":  1e9,  "bn":  1e9,  "b": 1e9,
    "million":  1e6,  "mn":  1e6,  "mm": 1e6, "m": 1e6,
    "thousand": 1e3,  "k":   1e3,
}

# Core regex - captures an optional $ / £ / €, the numeric part, optional
# scale word (billion / million / ...) and an optional trailing unit such
# as % / USD / employees / roles.  We keep the capture loose and clean up
# in _parse_match().
_NUM_RE = re.compile(
    r"""
    (?P<prefix>\$|£|€|USD\s|EUR\s|GBP\s)?          # optional currency
    (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?        # 12,345.67
               |\d+(?:\.\d+)?)                      # 1234.56 or 42
    \s*
    (?P<scale>trillion|billion|million|thousand     # scale word
              |trn|bn|mn|mm|tn
              |[TBMK](?![a-zA-Z]))?                  # bare-letter scale,
                                                     # allowed even when
                                                     # glued to digit
                                                     # (e.g. "$67B")
    \s*
    (?P<suffix>%                                     # percent
              |percent
              |percentage\s+points?
              |p\.?p\.?s?\b
              |basis\s+points?
              |bps?\b
              |USD|EUR|GBP
              |years?|months?|weeks?|days?|hours?
              |employees?|people|roles?|positions?
              |countries|states|regions?
              |customers?|users?
              |accounts?|projects?|products?
              )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Tokens that must NOT be treated as numeric values even if they look
# like numbers (years are a classic false-positive source).
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
# Page numbers and section IDs (common in running headers the cleaner missed)
_PAGE_HINT = re.compile(r"\bpage\s+$", re.IGNORECASE)


@dataclass
class NumericClaim:
    sentence:       str
    value:          float         # normalised magnitude
    unit:           str           # bucket: "usd", "percent", "count:role", "years", ...
    raw:            str           # original match text
    section_id:     int
    section_label:  str
    section_heading: str = ""
    tokens:         set = field(default_factory=set)   # content tokens for entity match
    anchors:        set = field(default_factory=set)   # strong anchor phrases
    topics:         set = field(default_factory=set)   # specific topic tokens
    scopes:         set = field(default_factory=set)   # scope words (global/total/…)
    years:          set = field(default_factory=set)   # year-scope mentioned
    is_pie:         bool = False                       # listed among pie-slice %s
    is_aggregate:   bool = False                       # whole-of-parts claim

    def as_display_value(self) -> str:
        return self.raw.strip()


def _normalise_unit(scale: str | None, suffix: str | None,
                    prefix: str | None, tail_context: str) -> str:
    """
    Return a canonical unit bucket. Values are only cross-compared if
    their unit buckets match exactly.
    """
    pfx = (prefix or "").strip().lower()
    sfx = (suffix or "").strip().lower().rstrip('s')
    scl = (scale  or "").strip().lower()

    # Currency wins if present
    if pfx in {"$", "usd"} or sfx in {"usd"}:
        return "usd"
    if pfx in {"£", "gbp"} or sfx in {"gbp"}:
        return "gbp"
    if pfx in {"€", "eur"} or sfx in {"eur"}:
        return "eur"

    # Percent vs pp vs bps are different enough to keep separate
    if sfx in {"%", "percent"}:
        return "percent"
    if sfx.startswith("percentage point") or sfx.startswith("p.p") or sfx == "pp":
        return "percentage_points"
    if sfx.startswith("basis point") or sfx.startswith("bp"):
        return "basis_points"

    if sfx in {"year", "month", "week", "day", "hour"}:
        return f"duration:{sfx}"

    # Count units
    COUNT_BUCKETS = {
        "employee": "count:employee",
        "role": "count:role", "position": "count:role",
        "people": "count:person",  "user": "count:user",
        "customer": "count:customer",
        "country": "count:country", "state": "count:state",
        "region": "count:region",
        "account": "count:account", "project": "count:project",
        "product": "count:product",
    }
    for k, v in COUNT_BUCKETS.items():
        if sfx.startswith(k):
            return v

    # Scale-only mentions ("1.45 trillion" / "4.7M") with no suffix —
    # disambiguate from the surrounding sentence so "4.7M unfilled AI
    # roles" becomes count:role rather than an unusable scale_only value.
    if scl and not sfx:
        tc = tail_context.lower()
        # Count-unit hints first (they're more specific than money hints
        # like "market" that appear in many count-unit sentences too).
        if any(w in tc for w in ("role", "position", "job ", " jobs",
                                 "opening", "vacanc")):
            return "count:role"
        if any(w in tc for w in ("employee", "headcount", "workforce",
                                 "staff", "hire", "hiring")):
            return "count:employee"
        if any(w in tc for w in (" people", "person ", "persons")):
            return "count:person"
        if any(w in tc for w in ("customer", "user ", "users",
                                 "subscriber", "account")):
            return "count:user"
        # Money hints
        if any(w in tc for w in ("dollar", "usd", "revenue", "spend", "spending",
                                 "investment", "market", "funding", "gdp",
                                 "valuation", "budget", "capex", "opex",
                                 "cost", "price")):
            return "usd"
        return "scale_only"

    # Anything else: treat as a unit-less numeric
    if scl:
        return "scale_only"
    return "unknown"


def _to_float(number_text: str, scale: str | None) -> float | None:
    try:
        v = float(number_text.replace(",", ""))
    except ValueError:
        return None
    if scale:
        key = scale.strip().lower()
        # Handle bare letters
        if key in {"t"}: key = "trillion"
        elif key in {"b"}: key = "billion"
        elif key in {"m"}: key = "million"
        elif key in {"k"}: key = "thousand"
        v *= _UNIT_SUFFIX.get(key, 1.0)
    return v


def _extract_values_from_sentence(
    sentence: str,
) -> list[tuple[float, str, str, int]]:
    """
    Return list of (value, unit_bucket, raw_text, match_start) tuples in
    the sentence. The match_start lets callers attach the nearest year to
    each value rather than the whole sentence's year set.

    Filters out year-like numbers and page hints.
    """
    out = []
    for m in _NUM_RE.finditer(sentence):
        number_text = m.group("number")
        if _YEAR_RE.match(number_text.replace(",", "")) and not m.group("prefix"):
            continue
        # Skip "Page 3" where "Page" immediately precedes the number
        before = sentence[max(0, m.start() - 8):m.start()]
        if _PAGE_HINT.search(before):
            continue
        value = _to_float(number_text, m.group("scale"))
        if value is None:
            continue
        # Ignore single-digit / tiny integers with no unit - they're usually
        # sentence numbering ("First, ... Second, ...") and generate noise.
        if value < 10 and not (m.group("scale") or m.group("prefix")
                               or m.group("suffix")):
            continue
        unit = _normalise_unit(
            m.group("scale"),
            m.group("suffix"),
            m.group("prefix"),
            sentence,
        )
        if unit == "unknown":
            continue
        # Drop tiny currency values with no scale suffix ("$3", "$5 per
        # dollar"): at enterprise-report granularity these are prices,
        # fees, or ratios — never macro market sizes — and they pollute
        # the pairing step with apples-to-oranges matches against $B/$T.
        if unit in {"usd", "gbp", "eur"} \
                and not m.group("scale") \
                and value < 1_000_000:
            continue
        # Also drop any currency figure whose sentence explicitly frames
        # it as a ratio ("$3 per dollar invested", "$5 per user", etc.),
        # regardless of scale.
        win_end = min(len(sentence), m.end() + 40)
        tail = sentence[m.end():win_end].lower()
        if unit in {"usd", "gbp", "eur"} and re.search(
                r"^\s*(?:[-–to0-9\. ]+)?\s*per\s+(?:dollar|\$|user|employee"
                r"|customer|capita|head|hour|month|year|day|seat|unit)",
                tail,
        ):
            continue
        out.append((value, unit, m.group(0).strip(), m.start()))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY / CONTENT-TOKEN MATCHING
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","and","or","but","of","to","for","in","on","at","with",
    "by","from","as","is","are","was","were","be","been","being","it","its",
    "this","that","these","those","their","they","them","has","have","had",
    "will","would","should","could","can","may","not","no","also","than",
    "then","so","if","while","which","who","our","we","your","you","his",
    "her","he","she","i","me","my","all","any","some","such","other","more",
    "most","less","least","section","report","figure","table","page","document",
    "about","approximately","around","roughly","nearly","over","under",
    "first","second","third",
    "including","across","throughout","within","according",
}
# NOTE: we deliberately keep "global", "year(s)", "market", "estimated",
# "total", "average", "new", "key" OUT of the stopwords — they frequently
# act as the semantic anchor in numeric claims ("global market",
# "total headcount", "annual revenue", etc.) and removing them starves
# the 3-shared-token attribution gate.

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{3,}")


def _content_tokens(text: str) -> set[str]:
    """Stemmed-lite content tokens for fuzzy entity matching."""
    toks = set()
    for m in _TOKEN_RE.findall(text.lower()):
        t = m.rstrip("s")
        if len(t) < 4 or t in _STOPWORDS:
            continue
        toks.add(t)
    return toks


# ─────────────────────────────────────────────────────────────────────────────
# ANCHOR PHRASES
# A "shared three content tokens" gate is too weak for business reports —
# words like "market" and "growth" appear on every page. We additionally
# require a shared *anchor phrase*: a specific multi-word proper noun
# ("Asia-Pacific", "AI market", "venture capital"), an acronym ("AI",
# "APAC", "CAGR"), or a known metric-noun bigram ("market share",
# "adoption rate", "enterprise spending").
# ─────────────────────────────────────────────────────────────────────────────

# Multi-word capitalised phrases: "North America", "Asia-Pacific",
# "AI Market", "Generative AI" (the second word is all-caps).
_MULTIWORD_ANCHOR_RE = re.compile(
    r'\b(?:'
    r'[A-Z][a-zA-Z]+(?:[ -][A-Z][a-zA-Z]+)+'      # "North America", "Asia-Pacific"
    r'|[A-Z][a-zA-Z]+(?:[ -][A-Z]{2,})'           # "Generative AI"
    r'|[A-Z]{2,}(?:[ -][A-Z][a-zA-Z]+)+'          # "EU AI Act", "APAC Market"
    r')\b'
)

# Stand-alone acronyms (≥2 uppercase letters).
_ACRONYM_RE = re.compile(r'\b([A-Z]{2,})\b')

# Acronyms that appear so generically they're not useful anchors.
_BLAND_ACRONYMS = {
    "AI", "US", "USA", "UK", "EU", "IT", "CEO", "CFO", "CTO",
    "H1", "H2", "Q1", "Q2", "Q3", "Q4",
}

# Domain bigrams: "X market/spending/revenue/rate/etc.".
_METRIC_NOUNS = {
    "market", "markets", "spending", "spend", "revenue", "revenues",
    "adoption", "growth", "investment", "investments", "funding", "valuation",
    "share", "rate", "cagr", "shortage", "headcount", "workforce",
    "headcounts", "talent", "roles", "hiring", "deployment", "deployments",
    "budget", "budgets", "pipeline", "pipelines", "maturity", "penetration",
    "capex", "opex",
    # Broader market-segment synonyms — "hardware sector", "generative
    # industry", "healthcare vertical", etc. all need to anchor.
    "sector", "sectors", "industry", "industries", "segment", "segments",
    "vertical", "verticals", "category", "categories", "ecosystem",
    "ecosystems", "platform", "platforms", "infrastructure",
    # Talent/workforce domain
    "position", "positions", "jobs", "openings", "vacancies",
    # Money/sizing
    "size", "sizes", "cost", "costs", "price", "prices", "fees", "loss",
    "losses",
}

# Two bigram shapes.  The first requires the left word to be ≥3 chars
# (rules out generic articles); the second explicitly allows a 2-char
# uppercase acronym on the left ("AI roles", "ML spending") because in
# an AI-heavy doc this IS the anchoring domain term.
_BIGRAM_ANCHOR_RE = re.compile(
    r'\b([a-zA-Z][a-zA-Z-]{2,})\s+(' + '|'.join(sorted(_METRIC_NOUNS)) + r')\b',
    re.IGNORECASE,
)
_ACRONYM_BIGRAM_RE = re.compile(
    r'\b([A-Z]{2,5})\s+(' + '|'.join(sorted(_METRIC_NOUNS)) + r')\b'
)

# Articles / determiners / fillers that sometimes start a "multiword
# capitalised" span because they're the first word of a sentence.
# Strip them so anchors like "the ai" or "our market" never form.
_MULTIWORD_LEFT_FILLER = {
    "the", "a", "an", "our", "their", "its", "this", "that", "these",
    "those", "his", "her", "my", "your",
}


def _anchor_phrases(text: str) -> set[str]:
    """
    Extract strong semantic anchors from a sentence. These are used to
    confirm that two numeric claims really talk about the same thing,
    beyond just surface-token overlap.
    """
    anchors: set[str] = set()

    # Multi-word capitalised phrases — but drop spans whose first word
    # is just a sentence-initial article/determiner ("The AI", "Our Market").
    for m in _MULTIWORD_ANCHOR_RE.finditer(text):
        phrase = m.group(0).strip()
        first, _, rest = phrase.partition(" ")
        if first.lower() in _MULTIWORD_LEFT_FILLER:
            phrase = rest.strip()
        phrase = phrase.lower()
        # Only keep it if it's still a *multi-word* phrase (space or hyphen).
        if len(phrase) >= 4 and (" " in phrase or "-" in phrase):
            anchors.add(phrase)

    # Acronyms (minus the generic ones)
    for m in _ACRONYM_RE.finditer(text):
        a = m.group(1)
        if a in _BLAND_ACRONYMS:
            continue
        anchors.add(a.lower())

    # Domain metric bigrams: "healthcare spending", "hardware sector",
    # "generative industry".  We store BOTH the full bigram AND the bare
    # qualifier (left word). This lets "hardware market" and "hardware
    # sector" both anchor to "hardware" while still keeping full bigrams
    # like "market share" useful when present on both sides.
    for m in _BIGRAM_ANCHOR_RE.finditer(text):
        left = m.group(1).lower().rstrip("s")
        right = m.group(2).lower().rstrip("s")
        if left in _STOPWORDS or len(left) < 3:
            continue
        anchors.add(f"{left} {right}")
        # Bare-qualifier anchor — but skip it if the qualifier is itself a
        # generic metric noun (e.g. "market revenue" → don't anchor on
        # "market" alone, that would light up half the document).
        if left not in _METRIC_NOUNS and left not in _GENERIC_QUALIFIERS:
            anchors.add(left)

    # Acronym + metric-noun bigrams ("AI roles", "ML spending"). These
    # are the anchoring domain terms in AI reports and would be lost by
    # the 3-char filter above. Bland acronyms ("AI", "US", etc.) are
    # still allowed here because the METRIC-NOUN right side
    # ("AI roles", "US market") narrows the meaning.
    for m in _ACRONYM_BIGRAM_RE.finditer(text):
        left = m.group(1).lower()
        right = m.group(2).lower().rstrip("s")
        anchors.add(f"{left} {right}")

    # "Qualifier AI noun" pattern ("healthcare AI market", "generative AI
    # spending"): the real anchor is the qualifier, not the bland "AI".
    for m in _QUALIFIER_AI_METRIC_RE.finditer(text):
        q = m.group(1).lower().rstrip("s")
        if q in _STOPWORDS or q in _GENERIC_QUALIFIERS or len(q) < 4:
            continue
        anchors.add(q)

    return anchors


# Qualifiers too generic to use as a standalone anchor — they'd light up
# half the document on their own. (They're still fine as part of a
# bigram anchor.)
_GENERIC_QUALIFIERS = {
    "global", "total", "overall", "annual", "major", "broad", "wider",
    "whole", "entire", "general", "main", "core", "new", "key",
    "estimated", "projected", "expected", "forecast", "forecasted",
    "current", "average", "typical", "recent", "future", "past",
    "worldwide", "domestic", "international",
    "enterprise",   # enterprise software/AI/etc is near-universal in these docs
    "business",
    "industry", "sector", "segment", "market", "company", "companies",
    "organization", "organizations", "organisation", "organisations",
    "firm", "firms",
}

# Match "<qualifier> AI <metric-noun>" (or ML/GenAI/LLM) so we can pull
# the qualifier out as the real anchor.
_QUALIFIER_AI_METRIC_RE = re.compile(
    r'\b([A-Za-z][A-Za-z-]{3,})\s+'
    r'(?:AI|ML|GenAI|LLM|LLMs|GPT)\s+'
    r'(' + '|'.join(sorted(_METRIC_NOUNS)) + r')\b',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# YEAR / TIME SCOPE
# Two claims about the same metric in *different* years are forecasts, not
# contradictions.  We skip the pair if both sentences specify years and
# those year-sets don't overlap.
# ─────────────────────────────────────────────────────────────────────────────

_YEAR_MENTION_RE = re.compile(r'\b((?:19|20)\d{2})\b')


def _years_in(text: str) -> set[int]:
    out: set[int] = set()
    for m in _YEAR_MENTION_RE.finditer(text):
        y = int(m.group(1))
        if 1980 <= y <= 2099:
            out.add(y)
    return out


def _nearest_years(sentence: str, pos: int, window: int = 25) -> set[int]:
    """
    Return the single year that sits closest to `pos` in `sentence`.

    In trajectory sentences like
    "grown from $8B in 2021 to $158B in 2025"
    sentence-wide year extraction wrongly attaches BOTH 2021 and 2025 to
    BOTH values. We pick the nearest (minimum character distance) year
    instead. A narrow window keeps the "$X in YEAR" pattern tightly
    coupled without catching the next clause's year.

    If no year sits inside the window, fall back to the full sentence
    year set so single-value sentences retain their year context.
    """
    candidates: list[tuple[int, int]] = []
    for m in _YEAR_MENTION_RE.finditer(sentence):
        y = int(m.group(1))
        if not (1980 <= y <= 2099):
            continue
        # Distance from the value (pos) to the nearest edge of the
        # year-match — whichever side the year is on.
        dist = min(abs(pos - m.start()), abs(pos - m.end()))
        candidates.append((dist, y))
    if not candidates:
        return set()
    # Keep only years that fall inside the window; if nothing inside,
    # fall back to the full sentence year set.
    inside = [y for d, y in candidates if d <= window]
    if inside:
        # Pick ALL years inside the window but prefer the closest — in
        # practice that usually means a single year.
        inside.sort(key=lambda y: min(
            (d for d, yy in candidates if yy == y), default=10**9))
        return {inside[0]}
    return _years_in(sentence)


def _years_compatible(a: set[int], b: set[int]) -> bool:
    """
    Compatible if either side is silent about years (meaning 'same period
    as the other'), or the year-sets overlap.
    """
    if not a or not b:
        return True
    return bool(a & b)


# ─────────────────────────────────────────────────────────────────────────────
# PIE-SLICE / STAT-LIST DETECTION
# Many false positives come from sentences like
#   "Asia-Pacific (28.1%), North America (41%), Europe (19%)"
# or  "data quality (68%), talent shortage (61%), integration (54%)".
# Those numbers are parts of a whole, not stand-alone metrics.
# ─────────────────────────────────────────────────────────────────────────────

_PAREN_PCT_RE    = re.compile(r'\([\s\d.]+%\)')                  # "(28.1%)"
_PCT_COUNT_RE    = re.compile(r'\b\d+(?:\.\d+)?\s*%')
_DOLLAR_COUNT_RE = re.compile(
    r'\$\s*\d[\d,.]*\s*(?:trillion|billion|million|bn|mn|t\b|b\b|m\b)?',
    re.IGNORECASE,
)


def _is_pie_sentence(text: str) -> bool:
    """
    True if the sentence is clearly listing shares of a whole or stacking
    several independent statistics.  Values inside pie/list sentences
    should not be cross-paired with values from other sentences.
    """
    # Two or more "(N%)" in the same sentence is the classic pie-slice
    # formatting.
    if len(_PAREN_PCT_RE.findall(text)) >= 2:
        return True

    # Three or more percentages in one sentence = ranked list of shares.
    if len(_PCT_COUNT_RE.findall(text)) >= 3:
        return True

    # Four or more dollar figures in one sentence = stat-stack, not an
    # assertion about a single metric.
    if len(_DOLLAR_COUNT_RE.findall(text)) >= 4:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY / LIST-LINE FILTER
# Reports often have a "key findings" header or a dense bullet-stat line
# that combines unrelated numbers ("72% Enterprise Adoption Key findings
# include ... dominance of North America ... healthcare as the leading
# vertical ..."). These are structurally not single claims and should not
# feed the numeric pairing logic at all.
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_TRIGGER_RE = re.compile(
    r'\b(?:key\s+findings?|key\s+points?|highlights?|at\s+a\s+glance'
    r'|major\s+findings?|executive\s+summary|takeaways?|in\s+summary'
    r'|we\s+project|in\s+our\s+view|forecast\s+outlook)\b',
    re.IGNORECASE,
)

# Aggregate / whole-vs-part claims: "three regions account for 90% of
# total market", "collectively represent 9.2%" — shouldn't pair with
# individual-component claims because they're describing sums vs parts.
_AGGREGATE_RE = re.compile(
    r'\b(?:collectively|combined|together\s+account|total\s+addressable'
    r'|aggregate|three\s+leading|account\s+for\s+(?:nearly|over|roughly)'
    r'|make\s+up|geographic(?:al(?:ly)?)?\s+concentration'
    r'|represent(?:ing)?\s+(?:over|nearly|approximately|roughly)'
    r'|three\s+major\s+regions)\b',
    re.IGNORECASE,
)


def _is_aggregate_claim(text: str) -> bool:
    """True if the sentence is a whole-of-parts aggregate claim."""
    return bool(_AGGREGATE_RE.search(text))


# Scope words that mark a claim as "the whole market" rather than a
# sub-segment.  Needed to block cross-scope pairs like "$391B global AI"
# vs "$62B healthcare AI" (total vs vertical) when the specific-topic
# side is healthcare but the other side is scoped globally.
_SCOPE_WORDS = {
    "global", "total", "overall", "worldwide", "entire", "whole",
    "aggregate", "combined", "all-up", "world-wide",
}


def _scope_tokens_in(text: str) -> set[str]:
    found: set[str] = set()
    for m in _TOKEN_RE.findall(text.lower()):
        if m in _SCOPE_WORDS:
            found.add(m)
    return found


def _paragraph_has_summary_trigger(text: str) -> bool:
    """True if ANY sentence in the paragraph is a summary/outlook line."""
    return bool(_SUMMARY_TRIGGER_RE.search(text))


def _paragraph_is_stat_dense(text: str) -> bool:
    """True if the whole paragraph has ≥6 numeric mentions (forecast block)."""
    num_count = (len(_PCT_COUNT_RE.findall(text))
                 + len(_DOLLAR_COUNT_RE.findall(text)))
    return num_count >= 6


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC TOKENS
# A curated set of "specific topic" words. If both sentences pick a topic
# token and those sets are disjoint, the pair is about different topics
# (e.g. healthcare-AI-market vs generative-AI-market) and should NOT be
# flagged as a contradiction even when the bigram "AI market" is shared.
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_TOKENS = {
    # verticals / industries
    "healthcare", "health", "medical", "pharmaceutical", "pharma",
    "retail", "banking", "finance", "financial", "insurance",
    "manufacturing", "automotive", "aerospace", "telecom",
    "telecommunications", "media", "entertainment", "education",
    "logistics", "transportation", "energy", "utilities",
    "government", "defense", "defence", "hospitality", "agriculture",
    "construction", "mining", "real-estate",
    # tech sub-categories
    "hardware", "software", "semiconductors", "semiconductor",
    "generative", "predictive", "analytics", "cybersecurity",
    "robotics", "automation", "cloud", "edge", "mobile",
    "consumer", "infrastructure",
    # talent / roles
    "unfilled", "vacant", "unstaffed",
    # geographies (standalone tokens; multi-word are caught by other rules)
    "apac", "emea", "nordics", "latam", "mena", "asean",
}


def _topic_tokens_in(text: str) -> set[str]:
    """Return the specific topic tokens present in a sentence."""
    found: set[str] = set()
    for m in _TOKEN_RE.findall(text.lower()):
        t = m.rstrip("s")
        if t in _TOPIC_TOKENS:
            found.add(t)
    return found


def _is_summary_line(text: str) -> bool:
    """True if the sentence looks like a dense summary/stat-stack line."""
    if _SUMMARY_TRIGGER_RE.search(text):
        return True
    # Number-dense sentences (4+ numeric mentions) are stat stacks, not
    # single-claim assertions. Lowered from 5 → 4 after observing that
    # 4-number forecast sentences like "$1.45T/30%/55%/38%" still leaked
    # false positives.
    num_count = (len(_PCT_COUNT_RE.findall(text))
                 + len(_DOLLAR_COUNT_RE.findall(text)))
    if num_count >= 4:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) >= 20]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_numeric_claims(chunks: list[dict]) -> list[NumericClaim]:
    """
    Walk every chunk and emit one NumericClaim per (sentence, numeric value).
    A sentence with three numbers produces three claims.

    Filters applied:
      * Skip summary/stat-stack lines (they aren't single assertions).
      * Skip sentences with fewer than 3 content tokens (no attribution).
      * Skip sentences with zero anchor phrases (nothing specific to anchor
        the claim to — prevents "at 12%" being paired with another "at 30%"
        just because both say "growth").
    """
    claims: list[NumericClaim] = []
    for chunk in chunks:
        label = chunk.get("label", "?")
        heading = chunk.get("heading", "")
        section_id = int(chunk.get("id", 0) or 0)
        chunk_text = chunk.get("text", "")

        # Paragraph-level filter: if the whole chunk is a forecast/outlook
        # block (any sentence has an outlook trigger, or the paragraph is
        # stat-dense), drop ALL sentences from it. This catches cases
        # where the trigger sits in sentence 1 but noisy numbers sit in
        # sentences 2-3 ("We project $1.45T... North America CAGR 28.5%
        # will trail Asia-Pacific's 36.1%...").
        if _paragraph_has_summary_trigger(chunk_text) \
                or _paragraph_is_stat_dense(chunk_text):
            continue

        for sent in _sentences(chunk_text):
            # Summary / stat-stack sentences produce mostly noise.
            if _is_summary_line(sent):
                continue
            values = _extract_values_from_sentence(sent)
            if not values:
                continue
            tokens = _content_tokens(sent)
            if len(tokens) < 3:
                continue  # not enough context to attribute the number
            anchors = _anchor_phrases(sent)
            if not anchors:
                continue  # no specific subject — don't risk a false pair
            topics = _topic_tokens_in(sent)
            scopes = _scope_tokens_in(sent)
            sentence_years = _years_in(sent)
            is_pie = _is_pie_sentence(sent)
            is_aggregate = _is_aggregate_claim(sent)
            # If there are multiple values AND multiple years in the sentence,
            # it's a trajectory ("from $8B in 2021 to $158B in 2025") and we
            # want the year nearest each value rather than the full set.
            use_nearest = len(values) >= 2 and len(sentence_years) >= 2
            for value, unit, raw, pos in values:
                if use_nearest:
                    years = _nearest_years(sent, pos)
                else:
                    years = sentence_years
                claims.append(NumericClaim(
                    sentence=sent,
                    value=value,
                    unit=unit,
                    raw=raw,
                    section_id=section_id,
                    section_label=label,
                    section_heading=heading,
                    tokens=tokens,
                    anchors=anchors,
                    topics=topics,
                    scopes=scopes,
                    years=years,
                    is_pie=is_pie,
                    is_aggregate=is_aggregate,
                ))
    return claims


def _relative_gap(a: float, b: float) -> float:
    """Symmetric relative gap as a fraction (0.0 = identical, 0.5 = 50% off)."""
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def _severity_from_gap(gap: float) -> tuple[str, int]:
    """Map (gap fraction) → (severity label, confidence pct)."""
    if gap >= 0.20:
        return "High", 97
    if gap >= 0.05:
        return "Medium", 88
    return "Low", 75


def find_numeric_contradictions(
    chunks: list[dict],
    min_shared_tokens: int = 1,
    min_shared_anchors: int = 1,
    min_gap: float = 0.02,
) -> list[dict]:
    """
    Return a list of contradiction dicts in the same shape as
    ``run_full_pipeline``. Each contradiction pairs two *sentences* that:

      * share ≥ ``min_shared_tokens`` content tokens,
      * share ≥ ``min_shared_anchors`` anchor phrases (proper-noun /
        metric-bigram overlap — ensures they talk about the *same thing*),
      * agree on year-scope (both silent, or at least one shared year),
      * are not both pie-slice / stat-stack sentences,
      * carry same-unit numeric values, and
      * disagree by more than ``min_gap`` fraction.

    These gates suppress the dominant false-positive class: two business
    percentages / dollar figures that happen to share generic tokens like
    "market" or "growth" but refer to unrelated metrics or different
    time horizons.
    """
    try:
        claims = extract_numeric_claims(chunks)
    except Exception as e:                      # fail-open
        logger.warning(f"numeric_extractor: extraction failed: {e}")
        return []

    # Bucket by unit first so we only compare like to like.
    buckets: dict[str, list[NumericClaim]] = defaultdict(list)
    for c in claims:
        buckets[c.unit].append(c)

    seen_pairs: set[tuple[int, int, int, int]] = set()
    contradictions: list[dict] = []

    for unit, group in buckets.items():
        if unit in ("unknown", "scale_only"):
            continue
        if len(group) < 2:
            continue

        for i in range(len(group)):
            a = group[i]
            for j in range(i + 1, len(group)):
                b = group[j]
                # Same-section pairs are almost always the same paragraph
                # telling a coherent story (outlook paragraph listing
                # 2028/2030 projections, or a range clause). Delegate
                # intra-section numeric contradictions to the NLI + LLM
                # intra-chunk checker rather than the numeric extractor.
                if a.section_id == b.section_id:
                    continue

                # ── Gate 0: drop all aggregate claims from pairing ──
                # Aggregate / "whole of parts" claims ("three regions
                # account for over 90% of total market", "collectively
                # represent 9.2%", "now represent approximately 40% of
                # total AI market revenue") cover heterogeneous
                # subsets that numeric extraction can't meaningfully
                # compare. Any genuine aggregate contradiction ("top-3
                # is 90%" vs "top-3 is 85%") will still be caught by
                # the NLI + LLM layer, which reads context.
                if a.is_aggregate or b.is_aggregate:
                    continue

                # ── Gate 1: content-token overlap ───────────────────
                if len(a.tokens & b.tokens) < min_shared_tokens:
                    continue

                # ── Gate 2: anchor-phrase overlap ───────────────────
                # A specific multi-word / acronym / metric-bigram must
                # appear in BOTH sentences. This is the core precision
                # improvement over the old "3 shared tokens" rule.
                if len(a.anchors & b.anchors) < min_shared_anchors:
                    continue

                # ── Gate 3: year-scope compatibility ────────────────
                if not _years_compatible(a.years, b.years):
                    continue

                # ── Gate 3b: topic-token compatibility ──────────────
                # If both sentences name specific topic tokens
                # (healthcare vs generative, hardware vs software,
                # retail vs banking, ...), require at least one in
                # common. This is what blocks "healthcare AI market
                # $62B" from pairing with "generative AI market $158B".
                if a.topics and b.topics and not (a.topics & b.topics):
                    continue

                # ── Gate 3c: scope mismatch ─────────────────────────
                # If one sentence is scoped "globally/total/worldwide"
                # and the other is scoped to a specific topic vertical
                # (healthcare, generative, retail, …), they are talking
                # about whole vs part and should not be paired.  Blocks
                # "$391B global AI market" from pairing with "$62B
                # healthcare AI spending" and similar.
                if (a.scopes and b.topics and not a.topics) or \
                   (b.scopes and a.topics and not b.topics):
                    continue

                # ── Gate 4: pie-slice suppression ───────────────────
                # Two sentences that are both listing pie-slice numbers
                # cannot contradict each other — they're inherently
                # describing different parts of a whole.
                if a.is_pie and b.is_pie:
                    continue

                # ── Gate 5: cross-table row subject mismatch ────────
                # Rows in two different tables share a numeric column
                # layout but describe completely different subjects
                # (e.g. "North America / $149B / 34% / ..."  vs
                #  "Corporate R&D / $41.8B / +35% / ..."). Tables use
                # their first cell as the row subject, which never
                # shows up as an anchor. Require two shared anchors
                # for cross-table pairs, not just one.
                a_is_row = (a.section_label or "").startswith("T") \
                    and ".R" in (a.section_label or "")
                b_is_row = (b.section_label or "").startswith("T") \
                    and ".R" in (b.section_label or "")
                if a_is_row and b_is_row \
                        and a.section_heading != b.section_heading \
                        and len(a.anchors & b.anchors) < 2:
                    continue

                gap = _relative_gap(a.value, b.value)
                if gap < min_gap:
                    continue

                # Tolerate "one quoting a range" patterns:
                #   A: "between $500K and $900K"
                #   B: "senior engineers earn $700K"
                # If B's value sits inside A's sentence numeric span, skip.
                if _value_within_range(a, b) or _value_within_range(b, a):
                    continue

                key = tuple(sorted([
                    (a.section_id, hash(a.sentence)),
                    (b.section_id, hash(b.sentence)),
                ]))
                flat = (key[0][0], key[0][1], key[1][0], key[1][1])
                if flat in seen_pairs:
                    continue
                seen_pairs.add(flat)

                severity, conf = _severity_from_gap(gap)
                contradictions.append(_build_contradiction_dict(a, b, gap, severity, conf))

    # Sort deterministic: highest severity, largest gap first
    sev_order = {"High": 0, "Medium": 1, "Low": 2}
    contradictions.sort(key=lambda c: (
        sev_order.get(c["severity"], 3), -c.get("numeric_gap", 0.0)
    ))
    return contradictions


def _value_within_range(inner: NumericClaim, outer: NumericClaim) -> bool:
    """True if inner.value lies inside a numeric range expressed in outer.sentence."""
    # Find *all* values in the outer sentence for the same unit.
    outer_vals = [t[0] for t in _extract_values_from_sentence(outer.sentence)
                  if t[1] == inner.unit]
    if len(outer_vals) < 2:
        return False
    lo, hi = min(outer_vals), max(outer_vals)
    # Small tolerance so 0.85 is "inside" a 0.80-0.90 range
    return lo * 0.98 <= inner.value <= hi * 1.02


def _build_contradiction_dict(
    a: NumericClaim,
    b: NumericClaim,
    gap: float,
    severity: str,
    confidence: int,
) -> dict:
    """Assemble the contradiction dict in the shape app.py/report expect."""
    explanation = (
        f"Both sentences reference the same metric ({a.unit.replace('_', ' ')}) "
        f"but give different figures: "
        f"{a.section_label} states \"{a.raw}\" while "
        f"{b.section_label} states \"{b.raw}\" "
        f"— a gap of {gap*100:.1f}%."
    )
    analysis = (
        f"CONTRADICTION FOUND:\n"
        f"Source: NUMERIC\n"
        f"Type: Numerical\n"
        f"Target Section: {a.section_label}\n"
        f"Conflicting Section: {b.section_label}\n"
        f"Statement A ({a.section_label}): \"{a.sentence}\"\n"
        f"Statement B ({b.section_label}): \"{b.sentence}\"\n"
        f"Explanation: {explanation}\n"
        f"Severity: {severity}\n"
        f"Confidence: {confidence}\n"
        f"Numeric Gap: {gap*100:.1f}%"
    )
    return {
        "target":           a.section_label,
        "target_display":   a.section_label,
        "target_text":      a.sentence,
        "analysis":         analysis,
        "related_sections": [b.section_label],
        "severity":         severity,
        "confidence":       confidence,
        "type":             "Numerical",
        "source":           "NUMERIC",   # distinguishes from NLI/LLM origin
        "numeric_gap":      gap,
        "verified":         True,        # deterministic, no LLM pass needed
    }
