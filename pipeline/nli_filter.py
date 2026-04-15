"""
NLI-Based Contradiction Pre-Filter  (v3 — calibrated)
======================================================

Uses a local cross-encoder NLI model (DeBERTa-v3-base) to score text
pairs for contradiction probability BEFORE sending to the LLM.

KEY INSIGHT (learned from testing):
  The NLI model's "contradiction" label (index 0) fires on BOTH genuine
  contradictions AND completely unrelated statements. This is standard
  NLI behaviour — "She likes pizza" vs "He went to the store" is treated
  as a mild contradiction because if one is the premise, the hypothesis
  doesn't follow.

  To fix this we use a COMBINED score:
      real_score = contradiction_prob × (1 − neutral_prob)
  This means pairs must be BOTH classified as contradictory AND NOT
  classified as neutral (i.e. they must be about the same topic).

  Additionally we require keyword overlap between the two statements
  as a fast pre-filter — two sentences about completely different
  topics cannot logically contradict each other.

Model: cross-encoder/nli-deberta-v3-base
  - Label order: [contradiction=0, entailment=1, neutral=2]
  - Runs on CPU in ~30-50 ms per pair
  - No API keys, no tokens, fully offline
"""

import re
import logging
import time
import numpy as np
from dataclasses import dataclass

from pipeline.precision_gate import accept as _precision_accept

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# UNIVERSE EXTRACTION (for the arithmetic reconciler)
# ─────────────────────────────────────────────────────────────────────

_UNIVERSE_NUM_RE = re.compile(
    r"""
    (?:\$\s?)?                                      # optional $
    -?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s?%)?
    | (?:\$\s?)?-?\d+(?:\.\d+)?(?:\s?%)?
    """,
    re.VERBOSE,
)


def extract_universe(chunks: list[dict]) -> list[float]:
    """
    Pull every numeric token out of the corpus to serve as the canonical
    value universe for the arithmetic reconciler. Deduplicated, sorted,
    year tokens (1900..2099) and trivial values (< 1) removed.
    """
    seen: set[float] = set()
    for ch in chunks:
        text = ch.get("text", "")
        for m in _UNIVERSE_NUM_RE.finditer(text):
            tok = m.group(0).replace("$", "").replace("%", "").replace(",", "").strip()
            try:
                v = float(tok)
            except ValueError:
                continue
            if abs(v) < 1.0:
                continue
            if 1900 <= v <= 2099 and v == int(v):
                continue                # bare years are not values
            seen.add(v)
    return sorted(seen)

# ─────────────────────────────────────────────────────────────────────────────
# LAZY MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

_nli_model = None
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Label indices (verified by calibration)
IDX_CONTRADICTION = 0
IDX_ENTAILMENT    = 1
IDX_NEUTRAL       = 2


def _softmax(logits):
    arr = np.array(logits, dtype=np.float64)
    exp = np.exp(arr - np.max(arr))
    return exp / exp.sum()


def get_nli_model():
    global _nli_model
    if _nli_model is None:
        print(f"[NLI] Loading model: {NLI_MODEL_NAME} ...")
        t0 = time.time()
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
        print(f"[NLI] Loaded in {time.time() - t0:.1f}s")

        # Quick sanity check
        test = _nli_model.predict([
            ("All employees must use VPN.", "VPN usage is optional."),
            ("She likes pizza.", "He went to the store."),
            ("The sky is blue.", "The sky is blue."),
        ])
        for i, (pair, pred) in enumerate(zip(
            ["contradiction", "neutral", "entailment"],
            test
        )):
            probs = _softmax(pred) if hasattr(pred, '__len__') else [pred]
            print(f"[NLI CHECK] {pair}: {[f'{p:.3f}' for p in probs]}")

    return _nli_model


# ─────────────────────────────────────────────────────────────────────────────
# TOPICAL OVERLAP (fast keyword filter)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "nor", "not", "no", "so", "yet", "both", "each",
    "all", "any", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "just", "only", "also", "if", "then", "that", "this",
    "it", "its", "their", "they", "them", "he", "she", "we", "our",
    "who", "which", "what", "when", "where", "how", "there", "here",
    "about", "up", "out", "down", "over", "own", "same", "while",
    # Meta/boilerplate words that appear in many sections but don't
    # indicate topical overlap — these cause massive false positives
    "statement", "statements", "section", "sections", "document",
    "documents", "documentation", "policy", "policies", "following",
    "contradictions", "contradiction", "contradicts", "commonly",
    "found", "inspired", "ambiguities", "common", "contains",
    "pairs", "mutually", "exclusive", "drawn", "realistic",
    "domains", "guidelines", "evaluation", "information",
    "conflicting", "reasoning", "consistency", "detection",
    "knowledge", "test", "purpose", "based", "including",
    "requirements", "required", "require", "ensures", "ensure",
    "according", "defined", "within", "without", "upon", "used",
    "using", "related", "regarding", "applicable", "appropriate",
    "necessary", "provided", "unless", "however", "therefore",
    "organization", "company", "employee", "employees", "management",
}


def _extract_keywords(text: str) -> set[str]:
    """Extract content words (non-stopwords, 3+ chars)."""
    words = set(re.findall(r'[a-z]{3,}', text.lower()))
    return words - _STOPWORDS


def topical_overlap(sent_a: str, sent_b: str, min_shared: int = 2) -> bool:
    """Check if two sentences share enough topic keywords."""
    kw_a = _extract_keywords(sent_a)
    kw_b = _extract_keywords(sent_b)
    shared = kw_a & kw_b
    return len(shared) >= min_shared


# Patterns that identify meta/boilerplate sentences (about the doc, not content)
_META_PATTERNS = [
    r'the following contradictions',
    r'contradictions? (?:are |is )?(?:commonly |frequently )?(?:found|embedded|inspired)',
    r'this (?:section|document|policy|report) (?:contains|defines|covers|describes)',
    r'purpose:?\s',
    r'evaluation of ',
    r'each section contains',
    r'mutually exclusive statements',
    r'statement [ab]\b',
    r'(?:contradicts?|conflicts? with) (?:statement )?[ab]\b',
]
_META_RE = re.compile('|'.join(_META_PATTERNS), re.IGNORECASE)


def is_meta_sentence(text: str) -> bool:
    """Return True if the sentence is about the document itself, not actual content."""
    return bool(_META_RE.search(text))


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

def split_into_statements(text: str, min_len: int = 20, filter_meta: bool = True) -> list[str]:
    """
    Split a chunk into individual statements.
    Rejoins PDF soft-wrapped lines, removes headings, splits on periods.
    Optionally filters out meta/boilerplate sentences.
    """
    # Rejoin soft-wrapped lines
    text = re.sub(r'(?<![.!?:])[ ]*\n(?=[a-z])', ' ', text)
    # Remove heading lines
    text = re.sub(r'^\d+[\.\d]*\s+[A-Z][^\n]{3,60}$', '', text, flags=re.MULTILINE)
    # Split on sentence-ending punctuation
    raw = re.split(r'(?<=[.!?])\s+', text)

    stmts = []
    for s in raw:
        s = s.strip().replace('\n', ' ')
        s = re.sub(r'\s{2,}', ' ', s)
        if len(s) >= min_len:
            if filter_meta and is_meta_sentence(s):
                continue  # Skip boilerplate
            stmts.append(s)
    return stmts


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContradictionCandidate:
    statement_a: str
    statement_b: str
    score: float                    # combined contradiction score 0-1
    source_chunk_label: str = ""
    target_chunk_label: str = ""
    is_intra: bool = False


def score_pairs(pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
    """
    Score pairs for contradiction using the COMBINED formula:
        score = contradiction_prob × (1 − neutral_prob)

    This ensures only topically-related contradictions score high.
    Unrelated pairs get high neutral_prob → score drops to ~0.
    """
    if not pairs:
        return []

    model = get_nli_model()
    predictions = model.predict(pairs, batch_size=batch_size)

    scores = []
    for pred in predictions:
        if hasattr(pred, '__len__') and len(pred) >= 3:
            probs = _softmax(list(pred))
            contra_p = float(probs[IDX_CONTRADICTION])
            neutral_p = float(probs[IDX_NEUTRAL])
            # Combined score: high contradiction AND low neutral
            combined = contra_p * (1.0 - neutral_p)
            scores.append(combined)
        else:
            scores.append(float(pred) if not hasattr(pred, '__len__') else float(pred[0]))

    # Debug
    if scores:
        above_05 = sum(1 for s in scores if s > 0.5)
        above_03 = sum(1 for s in scores if s > 0.3)
        top = max(scores)
        print(f"    scored {len(scores)} pairs | "
              f"max={top:.3f} | >0.5: {above_05} | >0.3: {above_03}")

    return scores


def find_intra_chunk_contradictions(
    chunk: dict,
    threshold: float = 0.40,
    max_candidates: int = 10,
    universe: list[float] | None = None,
) -> list[ContradictionCandidate]:
    """Find contradictory pairs WITHIN a single chunk."""
    statements = split_into_statements(chunk["text"])
    if len(statements) < 2:
        return []

    # Pre-filter: topical overlap → precision gate → only then NLI scoring.
    pairs = []
    pair_indices = []
    gate_drops = 0
    for i in range(len(statements)):
        for j in range(i + 1, len(statements)):
            if not topical_overlap(statements[i], statements[j], min_shared=1):
                continue
            decision = _precision_accept(
                statements[i], statements[j], universe=universe
            )
            if not decision.keep:
                gate_drops += 1
                logger.debug(
                    "precision_gate DROP intra[%s] stage=%s reason=%s",
                    chunk.get("label", "?"), decision.stage, decision.reason,
                )
                continue
            pairs.append((statements[i], statements[j]))
            pair_indices.append((i, j))

    if gate_drops:
        logger.info(
            "intra[%s]: precision_gate dropped %d pairs before NLI",
            chunk.get("label", "?"), gate_drops,
        )

    if not pairs:
        return []

    scores = score_pairs(pairs)

    candidates = []
    for score, (i, j) in zip(scores, pair_indices):
        if score >= threshold:
            candidates.append(ContradictionCandidate(
                statement_a=statements[i],
                statement_b=statements[j],
                score=score,
                source_chunk_label=chunk["label"],
                target_chunk_label=chunk["label"],
                is_intra=True,
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def find_cross_chunk_contradictions(
    target_chunk: dict,
    related_chunks: list[dict],
    threshold: float = 0.55,
    max_candidates: int = 10,
    universe: list[float] | None = None,
) -> list[ContradictionCandidate]:
    """
    Find contradictory pairs BETWEEN target and related chunks.
    Uses stricter threshold than intra-chunk because cross-section
    pairs are much noisier (different topics, meta text, etc.)
    """
    target_stmts = split_into_statements(target_chunk["text"])
    if not target_stmts:
        return []

    pairs = []
    meta = []
    gate_drops = 0

    for rc in related_chunks:
        related_stmts = split_into_statements(rc["text"])
        for i, ts in enumerate(target_stmts):
            for j, rs in enumerate(related_stmts):
                if not topical_overlap(ts, rs, min_shared=2):
                    continue
                decision = _precision_accept(ts, rs, universe=universe)
                if not decision.keep:
                    gate_drops += 1
                    logger.debug(
                        "precision_gate DROP cross[%s↔%s] stage=%s reason=%s",
                        target_chunk.get("label", "?"), rc.get("label", "?"),
                        decision.stage, decision.reason,
                    )
                    continue
                pairs.append((ts, rs))
                meta.append((rc["label"], ts, rs))

    if gate_drops:
        logger.info(
            "cross[%s]: precision_gate dropped %d pairs before NLI",
            target_chunk.get("label", "?"), gate_drops,
        )

    if not pairs:
        return []

    scores = score_pairs(pairs)

    candidates = []
    for score, (rc_label, ts, rs) in zip(scores, meta):
        if score >= threshold:
            candidates.append(ContradictionCandidate(
                statement_a=ts,
                statement_b=rs,
                score=score,
                source_chunk_label=target_chunk["label"],
                target_chunk_label=rc_label,
                is_intra=False,
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def deduplicate_candidates(candidates: list[ContradictionCandidate]) -> list[ContradictionCandidate]:
    """Remove duplicate pairs (same statements, possibly swapped order)."""
    seen: dict[tuple, ContradictionCandidate] = {}
    for c in candidates:
        key = tuple(sorted([_norm(c.statement_a), _norm(c.statement_b)]))
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    deduped = list(seen.values())
    deduped.sort(key=lambda c: c.score, reverse=True)
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

def prefilter_chunks(
    chunks: list[dict],
    retriever,
    embeddings,
    top_k: int = 5,
    alpha: float = 0.5,
    contradiction_threshold: float = 0.40,
    progress_callback=None,
) -> dict:
    """
    Run NLI pre-filtering on all chunks.
    Returns dict[chunk_index] → {intra_candidates, cross_candidates, ...}
    """
    t0 = time.time()
    results = {}
    all_intra = []
    all_cross = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks), "nli_scan", chunk["label"], t0)

        # Intra-chunk
        intra = find_intra_chunk_contradictions(
            chunk, threshold=contradiction_threshold
        )
        all_intra.extend(intra)

        # Cross-chunk
        related = retriever.retrieve(
            query_embedding=embeddings[i],
            query_text=chunk["text"],
            k=top_k,
            exclude_id=chunk["id"],
            alpha=alpha,
        )
        related = related or []

        cross = find_cross_chunk_contradictions(
            chunk, related, threshold=contradiction_threshold
        ) if related else []
        all_cross.extend(cross)

        results[i] = {
            "intra_candidates": intra,
            "cross_candidates": cross,
            "related_chunks": related,
            "needs_llm": len(intra) > 0 or len(cross) > 0,
        }

        n_stmts = len(split_into_statements(chunk["text"]))
        print(f"  [{chunk['label']}] stmts={n_stmts} "
              f"intra={len(intra)} cross={len(cross)}")

    # Global dedup
    all_intra = deduplicate_candidates(all_intra)
    all_cross = deduplicate_candidates(all_cross)

    # Rebuild with deduped
    for i, chunk in enumerate(chunks):
        label = chunk["label"]
        results[i]["intra_candidates"] = [
            c for c in all_intra if c.source_chunk_label == label
        ]
        results[i]["cross_candidates"] = [
            c for c in all_cross if c.source_chunk_label == label
        ]
        results[i]["needs_llm"] = (
            len(results[i]["intra_candidates"]) > 0 or
            len(results[i]["cross_candidates"]) > 0
        )

    elapsed = time.time() - t0
    total_intra = sum(len(r["intra_candidates"]) for r in results.values())
    total_cross = sum(len(r["cross_candidates"]) for r in results.values())
    flagged = sum(1 for r in results.values() if r["needs_llm"])
    print(f"\nNLI SUMMARY: {len(chunks)} chunks in {elapsed:.1f}s | "
          f"{flagged} flagged | {total_intra} intra + {total_cross} cross "
          f"(after dedup) | threshold={contradiction_threshold}")
    return results
