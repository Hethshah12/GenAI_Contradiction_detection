import os
import re
import time
import json
from groq import Groq
from dotenv import load_dotenv
from pipeline.hybrid_retriever import HybridRetriever

load_dotenv()

SYSTEM_PROMPT = """You are a strict document analyst. Detect ONLY clear logical contradictions in company reports.

You will receive a TARGET SECTION and several RELATED SECTIONS from the same document.

CONTRADICTION TYPES:
- DIRECT: Two sections state opposite facts (e.g., revenue increased vs revenue decreased)
- CONDITIONAL: A rule in one section conflicts with a condition stated elsewhere
- EXCEPTION-BASED: One section states a general rule, another creates an inconsistent exception

IMPORTANT RULES:
- Only flag a contradiction if you are at least 60% confident it is a genuine logical conflict.
- Do NOT flag: topic overlaps, different emphasis, unrelated content, disclaimers vs content, headers vs body text.
- Do NOT guess. If unsure, respond with NO CONTRADICTION FOUND.

IF you find a real contradiction, respond in EXACTLY this format, nothing else:
CONTRADICTION FOUND:
Type: [Direct / Conditional / Exception-based]
Target Section: [section label]
Conflicting Section: [section label]
Explanation: [2-3 sentences explaining the specific logical conflict]
Severity: [High / Medium / Low]
Confidence: [integer between 60 and 100]

Severity guide:
- High: contradicts financial figures, legal obligations, or core policy
- Medium: creates ambiguity in procedures or requirements
- Low: minor inconsistency in wording

IF there is no real contradiction, respond with EXACTLY this and nothing else:
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
    match = re.search(r'Type:\s*(Direct|Conditional|Exception-based)', result, re.IGNORECASE)
    return match.group(1) if match else "Direct"


def check_contradiction(target_chunk: dict, related_chunks: list, retries: int = 3) -> str:
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

    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                temperature=0.0,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(2 ** attempt * 10)
            else:
                return "NO CONTRADICTION FOUND"

    return "NO CONTRADICTION FOUND"


def _make_pair_key(id1: int, id2: int) -> str:
    return f"{min(id1, id2)}-{max(id1, id2)}"


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
    retriever = HybridRetriever(chunks, embeddings, index)

    contradictions = []
    clean_count = 0
    checked_pairs = set()
    heatmap_data = {}

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks), chunk["label"])

        related = retriever.retrieve(
            query_embedding=embeddings[i],
            query_text=chunk["text"],
            k=top_k,
            exclude_id=chunk["id"],
            alpha=alpha
        )

        if not related:
            clean_count += 1
            continue

        new_related = []
        for r in related:
            key = _make_pair_key(chunk["id"], r["id"])
            if key not in checked_pairs:
                checked_pairs.add(key)
                new_related.append(r)

        if not new_related:
            clean_count += 1
            continue

        if i > 0:
            time.sleep(delay_between_calls)

        result = check_contradiction(chunk, new_related)

        if not is_valid_contradiction(result):
            clean_count += 1
            continue

        confidence = parse_confidence(result)

        if confidence is None or confidence < min_confidence:
            clean_count += 1
            continue

        severity = parse_severity(result)
        c_type   = parse_contradiction_type(result)

        for r in new_related:
            heatmap_data[(chunk["label"], r["label"])] = confidence

        contradictions.append({
            "target":           chunk["label"],
            "target_display":   chunk.get("display", chunk["label"]),
            "target_text":      chunk["text"],
            "analysis":         result,
            "related_sections": [r["label"] for r in new_related],
            "severity":         severity,
            "confidence":       confidence,
            "type":             c_type,
        })

    return contradictions, clean_count, heatmap_data