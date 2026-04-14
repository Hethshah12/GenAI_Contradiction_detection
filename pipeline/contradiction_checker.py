import os
import re
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from dotenv import load_dotenv
from pipeline.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

load_dotenv()

SYSTEM_PROMPT = """You are a strict document analyst. Detect ALL clear logical contradictions in long-form documents (policies, reports, contracts, specifications).

You will receive a TARGET SECTION and optionally several RELATED SECTIONS from the same document.

YOUR TWO JOBS:
1. INTRA-SECTION: Find contradictions WITHIN the Target Section itself (statements that contradict each other in the same section).
2. CROSS-SECTION: Find contradictions BETWEEN the Target Section and any Related Section.

CONTRADICTION TYPES:
- DIRECT: Two statements state opposite facts (e.g., "X is allowed" vs "X is prohibited")
- CONDITIONAL: A rule conflicts with a condition stated elsewhere
- EXCEPTION-BASED: A general rule is followed by an inconsistent exception
- NUMERICAL: Conflicting figures, statistics, or quantities
- TEMPORAL: Timeline or date references are inconsistent
- DEFINITIONAL: The same term is defined or used differently
- SCOPE: One statement applies a rule globally, another limits it without acknowledgment

IMPORTANT RULES:
- You may find 0, 1, or MULTIPLE contradictions. Report ALL of them.
- Only flag if you are at least 60% confident it is a genuine logical conflict.
- Do NOT flag: topic overlaps, different emphasis, unrelated content, disclaimers vs content.

For EACH contradiction found, use EXACTLY this format:

CONTRADICTION FOUND:
Source: [INTRA or CROSS]
Type: [Direct / Conditional / Exception-based / Numerical / Temporal / Definitional / Scope]
Target Section: [section label]
Conflicting Section: [section label, or "same section" for intra]
Explanation: [2-3 sentences explaining the specific logical conflict]
Severity: [High / Medium / Low]
Confidence: [integer between 60 and 100]

---

(Separate multiple contradictions with ---)

Severity guide:
- High: contradicts financial figures, legal obligations, core policy, or numerical data
- Medium: creates ambiguity in procedures, requirements, or definitions
- Low: minor inconsistency in wording or scope

IF there are NO contradictions at all, respond with EXACTLY:
NO CONTRADICTION FOUND"""

INTRA_CHUNK_SYSTEM_PROMPT = """You are a strict document analyst specializing in detecting logical contradictions WITHIN a single section of text.

You will receive ONE section. Your job is to find statements within it that contradict each other.

CONTRADICTION TYPES:
- DIRECT: Two statements say opposite things (e.g., "X is allowed" followed by "X is prohibited")
- CONDITIONAL: A rule conflicts with a condition stated in the same section
- EXCEPTION-BASED: A general rule is followed by an inconsistent exception
- NUMERICAL: Conflicting numbers, percentages, or quantities within the section
- TEMPORAL: Inconsistent dates, timelines, or deadlines
- DEFINITIONAL: The same term is used with conflicting meanings
- SCOPE: A broad rule is followed by an unacknowledged narrower limitation

IMPORTANT RULES:
- Flag EACH distinct contradiction pair separately.
- Only flag genuine logical conflicts, not emphasis or nuance.
- You may find 0, 1, or MULTIPLE contradictions in a single section.

IF you find contradictions, respond with ALL of them, each in this EXACT format:

CONTRADICTION FOUND:
Type: [Direct / Conditional / Exception-based / Numerical / Temporal / Definitional / Scope]
Statement A: [exact or near-exact quote of the first statement]
Statement B: [exact or near-exact quote of the conflicting statement]
Explanation: [2-3 sentences explaining the specific logical conflict]
Severity: [High / Medium / Low]
Confidence: [integer between 60 and 100]

---

(Use --- to separate multiple contradictions.)

IF there are no contradictions, respond with EXACTLY:
NO CONTRADICTION FOUND"""


PARAM_SYSTEM_PROMPT = """You are a document analysis configuration expert. 
Given a sample of text from a document, determine the optimal RAG pipeline parameters.

Respond ONLY with a valid JSON object — no explanation, no markdown, no code fences.

Required JSON format:
{
  "top_k": <integer 2-8>,
  "alpha": <float 0.0-1.0>,
  "min_chunk": <integer 50-300>,
  "max_chunk": <integer 300-2000>,
  "api_delay": <float 1.0-5.0>,
  "min_confidence": <integer 60-95>,
  "reasoning": "<one sentence why>"
}

Parameter guidance:
- top_k: Number of sections to compare each chunk against. Use 3-4 for short/dense docs, 5-7 for long/complex ones.
- alpha: Balance between keyword (0.0) and semantic (1.0) retrieval. Use 0.3-0.4 for technical/legal docs with exact terms, 0.6-0.7 for narrative/report docs.
- min_chunk: Minimum chunk size in chars. Use 80-120 for dense docs, 150-200 for verbose docs.
- max_chunk: Maximum chunk size in chars. Use 500-800 for technical docs, 1000-1500 for narrative docs.
- api_delay: Seconds between API calls. Always use 2.0 (Groq free tier safe default).
- min_confidence: Minimum confidence % to flag. Use 65-70 for thorough detection, 80-85 for strict/high-precision."""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")
    return Groq(api_key=api_key)


def auto_detect_params(raw_text: str) -> dict:
    """
    Use LLM to auto-detect optimal pipeline parameters based on document content.
    Falls back to safe defaults if anything goes wrong.
    """
    defaults = {
        "top_k": 5,
        "alpha": 0.5,
        "min_chunk": 150,
        "max_chunk": 1000,
        "api_delay": 2.0,
        "min_confidence": 70,
        "reasoning": "Default parameters applied."
    }

    # Sample first ~2000 chars for param detection (cheap call)
    sample = raw_text[:2000].strip()
    if len(sample) < 100:
        return defaults

    try:
        client = get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": PARAM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this document sample and return optimal parameters:\n\n{sample}"}
            ],
            temperature=0.0,
            max_tokens=300
        )
        raw = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences
        raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        params = json.loads(raw)

        # Validate and clamp all values
        result = {
            "top_k":          max(2, min(8,    int(params.get("top_k", defaults["top_k"])))),
            "alpha":          max(0.0, min(1.0, float(params.get("alpha", defaults["alpha"])))),
            "min_chunk":      max(50, min(300,  int(params.get("min_chunk", defaults["min_chunk"])))),
            "max_chunk":      max(300, min(2000, int(params.get("max_chunk", defaults["max_chunk"])))),
            "api_delay":      2.0,  # Always 2.0 for Groq free tier safety
            "min_confidence": max(60, min(95,   int(params.get("min_confidence", defaults["min_confidence"])))),
            "reasoning":      str(params.get("reasoning", defaults["reasoning"]))[:200]
        }
        return result

    except Exception:
        return defaults


def is_valid_contradiction(result: str) -> bool:
    if "NO CONTRADICTION FOUND" in result:
        return False
    if "API error" in result or "max retries" in result:
        return False
    required = ["CONTRADICTION FOUND", "Type:", "Conflicting Section:", "Explanation:", "Severity:", "Confidence:"]
    return all(field in result for field in required)


def parse_confidence(result: str) -> int:
    match = re.search(r'Confidence:\s*(\d+)', result)
    if match:
        val = int(match.group(1))
        return min(100, max(0, val))
    return None


def parse_severity(result: str) -> str:
    match = re.search(r'Severity:\s*(High|Medium|Low)', result, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Low"


def parse_contradiction_type(result: str) -> str:
    match = re.search(
        r'Type:\s*(Direct|Conditional|Exception-based|Numerical|Temporal|Definitional|Scope)',
        result, re.IGNORECASE
    )
    return match.group(1) if match else "Direct"


def check_contradiction(target_chunk: dict, related_chunks: list, retries: int = 3) -> str:
    """
    Unified check: finds BOTH intra-section and cross-section contradictions
    in a single LLM call. Returns raw model output (may contain multiple
    CONTRADICTION FOUND blocks separated by ---).
    """
    related_text = ""
    for rc in related_chunks:
        related_text += f"\n\n[{rc['label']}]:\n{rc['text']}"

    user_message = (
        f"TARGET SECTION [{target_chunk['label']}]:\n"
        f"{target_chunk['text']}\n\n"
    )
    if related_chunks:
        user_message += f"RELATED SECTIONS:{related_text}\n\n"

    user_message += (
        "Find ALL logical contradictions:\n"
        "1) WITHIN the Target Section itself (intra-section)\n"
        "2) BETWEEN the Target Section and any Related Section (cross-section)\n"
        "Report every contradiction you find. Be strict — only flag genuine conflicts."
    )

    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                temperature=0.0,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(2 ** attempt * 10)
            else:
                return "NO CONTRADICTION FOUND"

    return "NO CONTRADICTION FOUND"


def parse_multiple_contradictions(raw: str) -> list[str]:
    """Split a raw LLM response into individual contradiction blocks."""
    if "NO CONTRADICTION FOUND" in raw and "CONTRADICTION FOUND:" not in raw:
        return []
    parts = re.split(r'\n-{3,}\n', raw)
    results = []
    for part in parts:
        part = part.strip()
        if "CONTRADICTION FOUND" in part and "Explanation:" in part:
            results.append(part)
    # If no separator was found but there's a single contradiction
    if not results and "CONTRADICTION FOUND" in raw and "Explanation:" in raw:
        results.append(raw.strip())
    return results


def parse_source_type(result: str) -> str:
    """Parse whether a contradiction is INTRA or CROSS."""
    match = re.search(r'Source:\s*(INTRA|CROSS)', result, re.IGNORECASE)
    return match.group(1).upper() if match else "CROSS"


def check_intra_chunk_contradictions(chunk: dict, retries: int = 3) -> list[str]:
    """
    Analyze a SINGLE chunk for internal contradictions.
    Returns a list of contradiction result strings (one per detected contradiction).
    """
    user_message = (
        f"SECTION [{chunk['label']}]:\n"
        f"{chunk['text']}\n\n"
        f"Find ALL logical contradictions within this single section. "
        f"Two statements that directly oppose each other = a contradiction."
    )

    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": INTRA_CHUNK_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                temperature=0.0,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()

            if "NO CONTRADICTION FOUND" in raw:
                return []

            # Split on --- separators to get individual contradictions
            parts = re.split(r'\n-{3,}\n', raw)
            results = []
            for part in parts:
                part = part.strip()
                if "CONTRADICTION FOUND" in part and "Explanation:" in part:
                    results.append(part)
            return results

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(2 ** attempt * 10)
            else:
                return []

    return []


def _make_pair_key(id1: int, id2: int) -> str:
    return f"{min(id1, id2)}-{max(id1, id2)}"


# ─────────────────────────────────────────────────────────────────────────────
# SHARED-CONCEPT GATE
# Rejects pairs of statements that have no meaningful noun / number overlap.
# NLI models happily mark unrelated sentences like "Japan is permissive"
# vs "Singapore is a regulatory leader" as contradictions because the
# surface structure looks contrastive. Requiring real shared referents
# kills that class of false positive cheaply (no extra API call).
# ─────────────────────────────────────────────────────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "for", "in", "on", "at",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "their", "they", "them",
    "has", "have", "had", "will", "would", "should", "could", "can", "may",
    "not", "no", "also", "than", "then", "so", "if", "while", "which", "who",
    "our", "we", "your", "you", "his", "her", "he", "she", "i", "me", "my",
    "all", "any", "some", "such", "other", "more", "most", "less", "least",
    "section", "report", "figure", "table", "page", "document",
}

_TOKEN_RE = re.compile(r"[A-Za-z]{4,}|\d+(?:[.,]\d+)*%?|\$[\d.,]+[BMK]?")


def _concept_tokens(text: str) -> set:
    """
    Extract the 'content words' and numeric tokens that anchor a
    statement's meaning. We keep:
      - words of length >= 4 that are not stopwords
      - raw numbers (optionally with %, commas, decimals)
      - dollar figures like $67, $1.45
    Proper nouns are preserved via case-insensitive match on the lemma.
    """
    tokens = set()
    for m in _TOKEN_RE.findall(text):
        t = m.lower().rstrip('s')  # crude singularisation
        if t in _STOPWORDS:
            continue
        tokens.add(t)
    return tokens


def _shared_concept_overlap(a: str, b: str) -> int:
    """Number of shared content/number tokens between two statements."""
    return len(_concept_tokens(a) & _concept_tokens(b))


def _pair_key_for_contradiction(c: dict) -> str:
    """Canonical undirected key for a contradiction pair."""
    a = c.get("target") or ""
    related = c.get("related_sections") or []
    b = (related[0] if related else a).replace(" (same section)", "")
    ids = sorted([a, b])
    return f"{ids[0]}__{ids[1]}"


def run_full_pipeline(
    chunks: list,
    embeddings,
    index,
    top_k: int = 5,
    delay_between_calls: float = 2.0,
    min_confidence: int = 60,
    alpha: float = 0.5,
    progress_callback=None
) -> tuple:
    """
    NLI-only contradiction detection pipeline (no API calls).
    Uses local DeBERTa model with combined scoring + topical overlap.

    progress_callback signature:
        fn(step: int, total: int, phase: str, label: str, t0: float)
    """
    from pipeline.nli_filter import prefilter_chunks

    retriever = HybridRetriever(chunks, embeddings, index)
    t0 = time.time()

    # ── NLI Detection (local, fast, zero API) ───────────────────────
    nli_results = prefilter_chunks(
        chunks=chunks,
        retriever=retriever,
        embeddings=embeddings,
        top_k=top_k,
        alpha=alpha,
        contradiction_threshold=0.40,
        progress_callback=progress_callback,
    )

    contradictions = []
    clean_count = 0
    heatmap_data = {}

    for i, chunk in enumerate(chunks):
        nli = nli_results.get(i, {})
        intra = nli.get("intra_candidates", [])
        cross = nli.get("cross_candidates", [])

        if not intra and not cross:
            clean_count += 1
            continue

        if progress_callback:
            progress_callback(i, len(chunks), "build_results", chunk["label"], t0)

        # Intra-chunk
        for c in intra:
            conf = int(c.score * 100)
            if conf < min_confidence:
                continue
            # Shared-concept gate: intra-section pairs still need at
            # least 2 shared content tokens - otherwise the chunk just
            # contains two unrelated sentences (e.g. "Japan permissive"
            # vs "Singapore regulatory leader").
            if _shared_concept_overlap(c.statement_a, c.statement_b) < 2:
                continue
            severity = "High" if conf >= 85 else ("Medium" if conf >= 70 else "Low")
            c_type = _classify_contradiction_type(c.statement_a, c.statement_b)

            heatmap_data[(chunk["label"], chunk["label"])] = conf

            analysis = (
                f"CONTRADICTION FOUND:\n"
                f"Source: INTRA\n"
                f"Type: {c_type}\n"
                f"Target Section: {chunk['label']}\n"
                f"Conflicting Section: {chunk['label']} (same section)\n"
                f"Statement A: \"{c.statement_a}\"\n"
                f"Statement B: \"{c.statement_b}\"\n"
                f"Explanation: These two statements within the same section "
                f"directly conflict. The first states \"{c.statement_a[:60]}...\" "
                f"while the second states \"{c.statement_b[:60]}...\", "
                f"creating a logical inconsistency.\n"
                f"Severity: {severity}\n"
                f"Confidence: {conf}\n"
                f"NLI Score: {c.score:.3f}"
            )

            contradictions.append({
                "target":           chunk["label"],
                "target_display":   chunk.get("display", chunk["label"]),
                "target_text":      chunk["text"],
                "analysis":         analysis,
                "related_sections": [chunk["label"] + " (same section)"],
                "severity":         severity,
                "confidence":       conf,
                "type":             c_type,
            })

        # Cross-chunk
        for c in cross:
            conf = int(c.score * 100)
            if conf < min_confidence:
                continue
            # Shared-concept gate (see intra comment).
            if _shared_concept_overlap(c.statement_a, c.statement_b) < 2:
                continue
            severity = "High" if conf >= 85 else ("Medium" if conf >= 70 else "Low")
            c_type = _classify_contradiction_type(c.statement_a, c.statement_b)

            heatmap_data[(chunk["label"], c.target_chunk_label)] = conf

            analysis = (
                f"CONTRADICTION FOUND:\n"
                f"Source: CROSS\n"
                f"Type: {c_type}\n"
                f"Target Section: {chunk['label']}\n"
                f"Conflicting Section: {c.target_chunk_label}\n"
                f"Statement A ({chunk['label']}): \"{c.statement_a}\"\n"
                f"Statement B ({c.target_chunk_label}): \"{c.statement_b}\"\n"
                f"Explanation: Statement in {chunk['label']} — "
                f"\"{c.statement_a[:60]}...\" — contradicts a statement in "
                f"{c.target_chunk_label} — \"{c.statement_b[:60]}...\"\n"
                f"Severity: {severity}\n"
                f"Confidence: {conf}\n"
                f"NLI Score: {c.score:.3f}"
            )

            contradictions.append({
                "target":           chunk["label"],
                "target_display":   chunk.get("display", chunk["label"]),
                "target_text":      chunk["text"],
                "analysis":         analysis,
                "related_sections": [c.target_chunk_label],
                "severity":         severity,
                "confidence":       conf,
                "type":             c_type,
            })

    # ── Pair deduplication ─────────────────────────────────────────
    # Several (A -> B) and (B -> A) entries frequently survive NLI
    # scoring when both chunks retrieve each other. Keep only the
    # highest-confidence entry per undirected pair.
    if contradictions:
        dedup: dict[str, dict] = {}
        for c in contradictions:
            key = _pair_key_for_contradiction(c)
            if key not in dedup or c["confidence"] > dedup[key]["confidence"]:
                dedup[key] = c
        # Preserve original ordering (stable by first-seen insertion)
        seen_keys = []
        for c in contradictions:
            k = _pair_key_for_contradiction(c)
            if k not in seen_keys:
                seen_keys.append(k)
        contradictions = [dedup[k] for k in seen_keys]

    # Optional LLM enrichment for top 3
    try:
        top3 = sorted(contradictions, key=lambda x: x["confidence"], reverse=True)[:3]
        for idx, cont in enumerate(top3):
            if progress_callback:
                progress_callback(idx, min(3, len(top3)), "llm_enrich", cont["target"], t0)
            if idx > 0:
                time.sleep(delay_between_calls)
            try:
                llm_exp = _get_llm_explanation(cont)
                if llm_exp:
                    cont["analysis"] += f"\n\nLLM Analysis:\n{llm_exp}"
            except Exception:
                pass
    except Exception:
        pass

    return contradictions, clean_count, heatmap_data


def _classify_contradiction_type(stmt_a: str, stmt_b: str) -> str:
    """
    Heuristic classification of contradiction type from statement text.
    """
    a_low = stmt_a.lower()
    b_low = stmt_b.lower()
    combined = a_low + " " + b_low

    # Numerical
    if re.search(r'\d+[%$]?\s', combined):
        nums_a = set(re.findall(r'\d+', a_low))
        nums_b = set(re.findall(r'\d+', b_low))
        if nums_a and nums_b and nums_a != nums_b:
            return "Numerical"

    # Temporal
    temporal_words = ["before", "after", "daily", "weekly", "monthly", "quarterly",
                      "immediately", "next quarter", "days", "hours"]
    if any(w in combined for w in temporal_words):
        return "Temporal"

    # Scope
    scope_words = ["all", "every", "never", "always", "no ", "any ", "only"]
    a_has_scope = any(w in a_low for w in scope_words)
    b_has_scope = any(w in b_low for w in scope_words)
    if a_has_scope and b_has_scope:
        # Check for opposing scope (e.g., "all" vs "not")
        if ("not " in a_low or "not " in b_low or
                "prohibited" in combined or "never" in combined):
            return "Scope"

    # Exception-based
    exception_words = ["unless", "except", "if ", "may ", "however"]
    if any(w in combined for w in exception_words):
        return "Exception-based"

    # Conditional
    if "if " in combined or "when " in combined or "provided " in combined:
        return "Conditional"

    # Direct (default for clear oppositions)
    direct_pairs = [
        ("must", "must not"), ("allowed", "prohibited"), ("required", "optional"),
        ("mandatory", "optional"), ("permitted", "not permitted"),
        ("can ", "cannot"), ("may ", "may not"), ("encouraged", "prohibited"),
        ("must", "not required"), ("strictly", "flexible"),
    ]
    for pos, neg in direct_pairs:
        if (pos in a_low and neg in b_low) or (neg in a_low and pos in b_low):
            return "Direct"

    # If none matched, still likely Direct
    return "Direct"


def _get_llm_explanation(contradiction: dict) -> str:
    """Get a brief LLM-generated explanation for a contradiction."""
    try:
        analysis = contradiction["analysis"]
        # Extract the two statements
        stmt_a_match = re.search(r'Statement A[^"]*"([^"]+)"', analysis)
        stmt_b_match = re.search(r'Statement B[^"]*"([^"]+)"', analysis)
        if not stmt_a_match or not stmt_b_match:
            return ""

        response = get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    f"In 2 sentences, explain the logical contradiction between:\n"
                    f"A: \"{stmt_a_match.group(1)}\"\n"
                    f"B: \"{stmt_b_match.group(1)}\"\n"
                    f"Be specific about the conflict."
                )
            }],
            temperature=0.0,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# SELF-CONSISTENCY SCORING (Novel)
# ─────────────────────────────────────────────────────────────────────────────
# Run the same contradiction check N times at different temperatures.
# Only keep results where ≥ K out of N runs agree.  This gives an
# empirically-grounded "stability score" instead of relying solely on
# the LLM's self-reported confidence.
# ─────────────────────────────────────────────────────────────────────────────

def check_with_self_consistency(
    target_chunk: dict,
    related_chunks: list,
    n_runs: int = 3,
    temperatures: list = None,
    agreement_threshold: int = 2,
) -> dict:
    """
    Run the contradiction check `n_runs` times at different temperatures
    and compute a stability score based on agreement.

    Args:
        target_chunk:        the chunk being tested
        related_chunks:      retrieved neighbours
        n_runs:              how many runs (default 3)
        temperatures:        list of temps to use (default [0.0, 0.3, 0.5])
        agreement_threshold: minimum agreeing runs to accept

    Returns:
        dict with keys:
          - is_contradiction: bool
          - stability_score:  float 0.0–1.0
          - best_result:      str (the result from the majority)
          - votes:            list of per-run results
    """
    if temperatures is None:
        temperatures = [0.0, 0.3, 0.5]

    # Pad or trim temperatures to match n_runs
    temps = (temperatures * ((n_runs // len(temperatures)) + 1))[:n_runs]

    related_text = ""
    for rc in related_chunks:
        related_text += f"\n\n[{rc['label']}]:\n{rc['text']}"

    user_message = (
        f"TARGET SECTION [{target_chunk['label']}]:\n"
        f"{target_chunk['text']}\n\n"
        f"RELATED SECTIONS:{related_text}\n\n"
        f"Does the TARGET SECTION logically contradict any RELATED SECTION? "
        f"Be strict — only flag genuine contradictions."
    )

    votes = []
    for temp in temps:
        try:
            response = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temp,
                max_tokens=400,
            )
            result = response.choices[0].message.content.strip()
            votes.append({
                "result": result,
                "is_contradiction": is_valid_contradiction(result),
                "temperature": temp,
            })
            time.sleep(1)  # rate-limit safety
        except Exception:
            votes.append({
                "result": "NO CONTRADICTION FOUND",
                "is_contradiction": False,
                "temperature": temp,
            })

    # Count agreements
    contradiction_votes = sum(1 for v in votes if v["is_contradiction"])
    no_contradiction_votes = len(votes) - contradiction_votes

    is_contradiction = contradiction_votes >= agreement_threshold
    stability_score = max(contradiction_votes, no_contradiction_votes) / len(votes)

    # Pick the best result (from contradiction votes if accepted, else from clean votes)
    if is_contradiction:
        best = next(v["result"] for v in votes if v["is_contradiction"])
    else:
        best = next(
            (v["result"] for v in votes if not v["is_contradiction"]),
            "NO CONTRADICTION FOUND"
        )

    return {
        "is_contradiction": is_contradiction,
        "stability_score":  round(stability_score, 2),
        "best_result":      best,
        "votes":            votes,
        "agreement":        f"{max(contradiction_votes, no_contradiction_votes)}/{len(votes)}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM VERIFIER (parallelized, cached, fail-open)
# ─────────────────────────────────────────────────────────────────────────────
# Second-pass verification for NLI-origin contradictions with uncertain
# confidence (60-85). Sends a tight, structured "YES/NO contradict?" prompt
# to Groq with self-consistency at T=[0.0, 0.3] — if both votes say NO we
# drop the pair; otherwise we keep it and bump confidence by a small
# verification bonus.
#
# Design constraints (user requirement — "don't ruin performance"):
#   * Only pairs in the uncertain band are verified. NUMERIC-origin and
#     confidence>=86 pairs are trusted and skipped.
#   * Parallel via ThreadPoolExecutor (default 4 workers). Groq is I/O-bound
#     so threads are fine despite the GIL.
#   * Per-statement-pair cache keyed by (stmt_a, stmt_b) hash — identical
#     pairs (rare but possible with symmetric re-detection) cost one call.
#   * Any exception is swallowed and the pair is retained unchanged
#     (fail-open: the verifier can never DELETE content the NLI+LLM stack
#     produced, only drop things it's confident are false positives).
#   * Short max_tokens (30) → ~100-150ms per call, 4x parallel → ~50ms/pair
#     amortised for a typical 10-pair document.
# ─────────────────────────────────────────────────────────────────────────────

_VERIFIER_SYSTEM = (
    "You are a strict fact-checker. You will be shown two statements "
    "pulled from a single document. Your job is to decide whether they "
    "are GENUINELY contradictory — they cannot both be true at the same "
    "time about the same subject.\n\n"
    "Only answer YES if they are a real logical conflict. Answer NO for "
    "topic overlap, different emphasis, one general + one specific "
    "(unless mutually exclusive), different time periods, or unrelated "
    "claims that just share vocabulary.\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: YES|NO\n"
    "REASON: <one short sentence>"
)

_VERIFY_UNCERTAIN_MIN = 60   # ignore things below the pipeline's min_confidence
_VERIFY_UNCERTAIN_MAX = 85   # >= 86 we trust the NLI+LLM-enrich pipeline
_VERIFY_BONUS_KEEP    = 3    # small confidence bump on a YES verdict
_VERIFY_PENALTY       = 10   # confidence drop if one vote disagrees (soft keep)


def _extract_statements_from_analysis(analysis: str) -> tuple[str, str]:
    """Pull out Statement A / Statement B quoted text from the analysis blob."""
    a = re.search(r'Statement A[^"]*"([^"]+)"', analysis or "")
    b = re.search(r'Statement B[^"]*"([^"]+)"', analysis or "")
    return (a.group(1).strip() if a else "",
            b.group(1).strip() if b else "")


def _verify_one_pair(stmt_a: str, stmt_b: str, temperature: float) -> tuple[bool, str]:
    """
    Single Groq call. Returns (is_contradiction_vote, reason_text).
    Raises on hard failure so the caller can treat it as "unknown".
    """
    client = get_client()
    user = (
        f"Statement A: \"{stmt_a}\"\n"
        f"Statement B: \"{stmt_b}\"\n\n"
        "Do these statements contradict each other?"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _VERIFIER_SYSTEM},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=60,
    )
    text = (response.choices[0].message.content or "").strip()
    verdict_match = re.search(r'VERDICT:\s*(YES|NO)', text, re.IGNORECASE)
    reason_match = re.search(r'REASON:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    is_yes = bool(verdict_match and verdict_match.group(1).upper() == "YES")
    reason = reason_match.group(1).strip() if reason_match else text[:160]
    return is_yes, reason


def _verify_with_self_consistency(stmt_a: str, stmt_b: str) -> dict:
    """
    Two-vote self-consistency check at T=[0.0, 0.3].
    Returns: {yes_votes, no_votes, reason, errored}.
    On any API failure we set errored=True so the caller keeps the pair.
    """
    votes = []
    reason = ""
    errored = False
    for temp in (0.0, 0.3):
        try:
            is_yes, r = _verify_one_pair(stmt_a, stmt_b, temp)
            votes.append(is_yes)
            if is_yes and not reason:
                reason = r
        except Exception as e:
            errored = True
            logger.debug(f"verifier call failed (T={temp}): {e}")
            # Don't retry inline — we're in parallel, fail-open.
            break
    yes = sum(1 for v in votes if v)
    no = sum(1 for v in votes if not v)
    return {"yes_votes": yes, "no_votes": no, "reason": reason, "errored": errored}


def verify_contradictions_with_llm(
    contradictions: list[dict],
    parallel: int = 4,
    progress_callback=None,
) -> list[dict]:
    """
    Second-pass LLM verification for uncertain NLI-origin contradictions.

    Args
    ----
    contradictions :   list of contradiction dicts from run_full_pipeline
                       (and/or numeric_extractor).
    parallel :         max concurrent API calls (default 4).
    progress_callback: optional fn(done, total, label) for UI.

    Returns
    -------
    list[dict] : same shape, with these fields possibly set on each entry:
        - ``verified``: True/False (True means passed LLM check)
        - ``verification_reason``: str (filled on YES verdict)
        - ``confidence``: adjusted up (+3) or down (-10) by vote outcome
        - Entries where BOTH votes said NO are DROPPED.
        - Entries where the verifier errored out are kept untouched.

    Non-goal
    --------
    The verifier never creates new contradictions. It only filters / scores
    existing ones. NUMERIC-origin pairs are now gated alongside NLI/LLM
    pairs when their confidence falls in the uncertain band.
    """
    if not contradictions:
        return contradictions

    # Partition: verify the uncertain middle band regardless of source.
    # NUMERIC pairs used to skip verification, but they turned out to be
    # the biggest false-positive generator on business reports (any two
    # percentages or dollar figures with enough shared vocabulary paired
    # up). Running them through the LLM gate is the single biggest
    # precision win available without sacrificing recall.
    #
    # We still skip:
    #   * confidence >= 86     — trust the upstream signal
    #   * confidence <  60     — already below the pipeline's min threshold
    #   * already LLM-verified — don't re-verify
    to_verify: list[tuple[int, dict, str, str]] = []
    for idx, c in enumerate(contradictions):
        if c.get("llm_verified") is True:
            continue  # already passed a previous LLM check
        conf = int(c.get("confidence", 0) or 0)
        if conf < _VERIFY_UNCERTAIN_MIN or conf > _VERIFY_UNCERTAIN_MAX:
            continue
        stmt_a, stmt_b = _extract_statements_from_analysis(c.get("analysis", ""))
        if not stmt_a or not stmt_b:
            continue  # nothing to ask about
        to_verify.append((idx, c, stmt_a, stmt_b))

    if not to_verify:
        return contradictions

    # Local cache to dedup identical statement pairs across the batch.
    cache: dict[tuple[str, str], dict] = {}
    results: dict[int, dict] = {}

    def _worker(task):
        idx, _c, a, b = task
        key = (a[:200], b[:200])
        if key in cache:
            return idx, cache[key]
        try:
            verdict = _verify_with_self_consistency(a, b)
        except Exception as e:
            logger.debug(f"verifier worker crashed: {e}")
            verdict = {"yes_votes": 0, "no_votes": 0, "reason": "", "errored": True}
        cache[key] = verdict
        return idx, verdict

    total = len(to_verify)
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as pool:
            futures = [pool.submit(_worker, task) for task in to_verify]
            for fut in as_completed(futures):
                try:
                    idx, verdict = fut.result(timeout=45)
                    results[idx] = verdict
                except Exception as e:
                    logger.debug(f"verifier future failed: {e}")
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total, "verify")
                    except Exception:
                        pass
    except Exception as e:
        # Catastrophic pool failure → skip verification entirely (fail-open).
        logger.warning(f"verifier pool failed, keeping all pairs: {e}")
        return contradictions

    out: list[dict] = []
    dropped = 0
    for idx, c in enumerate(contradictions):
        v = results.get(idx)
        if v is None:
            out.append(c)
            continue
        if v.get("errored") or (v["yes_votes"] + v["no_votes"]) == 0:
            # Verifier couldn't give a verdict — keep the pair untouched.
            out.append(c)
            continue

        # 2 YES → definite keep + bump, 1-1 → soft-keep with penalty,
        # 2 NO → drop.
        if v["yes_votes"] >= 2:
            c = dict(c)  # shallow copy so we don't mutate caller state
            c["verified"] = True
            c["llm_verified"] = True
            c["verification_reason"] = v.get("reason", "")
            c["confidence"] = min(99, int(c.get("confidence", 0)) + _VERIFY_BONUS_KEEP)
            if v.get("reason"):
                c["analysis"] = (c.get("analysis", "") +
                                 f"\n\nLLM Verification: {v['reason']}")
            out.append(c)
        elif v["yes_votes"] == 1:
            c = dict(c)
            c["verified"] = False
            c["llm_verified"] = False
            c["confidence"] = max(60, int(c.get("confidence", 0)) - _VERIFY_PENALTY)
            out.append(c)
        else:
            # Both votes NO → drop.
            dropped += 1
            continue

    if dropped:
        logger.info(f"LLM verifier dropped {dropped}/{total} uncertain pairs")

    return out