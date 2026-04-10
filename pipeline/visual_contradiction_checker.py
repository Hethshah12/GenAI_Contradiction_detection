"""
Visual-Textual Contradiction Checker
=====================================

Detects contradictions between visual elements (charts, graphs, tables)
and textual claims within the SAME document.  This is the core novelty
of the project — no prior published work specifically addresses
within-document visual-textual consistency checking.

Contradiction Taxonomy (Visual-Textual)
---------------------------------------
1. NUMERICAL   — Chart shows value X, text claims Y  (e.g., chart: 42%, text: "over 60%")
2. TREND       — Chart shows decline, text says "growth"
3. SCOPE       — Chart covers Q1-Q3, text draws "full-year" conclusions
4. CATEGORICAL — Chart labels don't match text's category references
5. TEMPORAL    — Chart's time range conflicts with text's temporal claims
6. OMISSION    — Text makes claims about data not present in the chart

Architecture
------------
For each chart chunk, retrieve the most relevant text chunks using the
existing hybrid retriever (FAISS + BM25).  Then ask the text LLM to
cross-reference the chart's structured description against the text,
looking specifically for the contradiction types above.

This is a SECOND-PASS analysis that runs AFTER the standard text-text
contradiction detection, providing a complementary layer of consistency
checking.

References
----------
- ChartCheck (ACL 2024)   — fact-checking claims against charts
- M3D (EMNLP 2024)        — multimodal inconsistency detection
- ContraDoc (NAACL 2024)   — text contradiction detection in documents
"""

import os
import re
import time
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

VISUAL_TEXTUAL_SYSTEM_PROMPT = """You are a multimodal document consistency analyst.
You specialise in detecting contradictions between CHARTS/GRAPHS and TEXTUAL CLAIMS
in the SAME document.

You will receive:
- A CHART DESCRIPTION (structured data extracted from a visual chart/graph)
- One or more TEXT SECTIONS from the same document

Your task: determine if ANY textual claim CONTRADICTS the chart data.

CONTRADICTION TYPES (visual-textual):
1. NUMERICAL:    Text states a number/percentage that conflicts with chart data
                 (e.g., chart shows revenue = $4.2M, text claims "revenue exceeded $5M")
2. TREND:        Text describes a trend that conflicts with chart pattern
                 (e.g., chart shows declining sales, text claims "sales grew consistently")
3. SCOPE:        Text draws conclusions beyond what the chart data covers
                 (e.g., chart shows Q1-Q3 data, text says "the full year showed...")
4. CATEGORICAL:  Text references categories/labels not present or different in chart
                 (e.g., chart has 4 product categories, text discusses a 5th)
5. TEMPORAL:     Text's time references conflict with chart's time period
                 (e.g., chart labeled "2023", text discusses it as "2024 figures")
6. OMISSION:     Text makes specific claims about data NOT shown in the chart
                 (e.g., text says "as shown in Figure 3, costs decreased" but chart shows no cost data)

IMPORTANT RULES:
- Only flag GENUINE contradictions where the chart data and text are logically incompatible.
- Approximate values are OK — flag only when differences are significant (>10% relative).
- Do NOT flag: rounding differences, stylistic paraphrasing, vague references.
- The chart description is an AI extraction and may have errors. If the chart data seems
  unreliable (low confidence), account for this in your assessment.

IF you find a contradiction, respond in EXACTLY this format:
VISUAL-TEXTUAL CONTRADICTION FOUND:
Type: [Numerical / Trend / Scope / Categorical / Temporal / Omission]
Chart Reference: [chart label/title]
Text Section: [section label]
Chart Claim: [what the chart data shows]
Text Claim: [what the text states]
Explanation: [2-3 sentences explaining the specific conflict]
Severity: [High / Medium / Low]
Confidence: [integer 60-100]

Severity guide:
- High: numerical values off by >20%, trend direction completely wrong, major scope mismatch
- Medium: moderate numerical differences (10-20%), partial trend mismatch, minor scope issues
- Low: borderline cases, possible rounding issues, ambiguous references

IF there is no real contradiction, respond with EXACTLY:
NO VISUAL-TEXTUAL CONTRADICTION FOUND"""


MULTI_HOP_VISUAL_PROMPT = """You are analysing potential multi-hop contradictions that
involve BOTH visual (chart/graph) and textual elements in a document.

A multi-hop contradiction occurs when:
- Section A makes claim X (text)
- Chart B shows data Y
- Section C references both A and B but draws a conclusion Z that is inconsistent
  with the combination of X and Y

You will receive a CHART DESCRIPTION, a PRIMARY TEXT SECTION, and one or more
SECONDARY TEXT SECTIONS that are related to both.

Determine if there is a multi-hop inconsistency — where the combination of
the chart data and one text section contradicts another text section.

IF found, respond in this format:
MULTI-HOP VISUAL CONTRADICTION FOUND:
Type: [Numerical / Trend / Scope / Categorical / Temporal]
Hop Chain: [Chart → Section A → Section B]
Explanation: [3-4 sentences tracing the logical chain]
Severity: [High / Medium / Low]
Confidence: [integer 60-100]

IF not found:
NO MULTI-HOP VISUAL CONTRADICTION FOUND"""


# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_visual_contradiction(result: str) -> dict | None:
    """
    Parse the LLM response into a structured contradiction dict.
    Returns None if no contradiction was found or parsing fails.
    """
    if "NO VISUAL-TEXTUAL CONTRADICTION FOUND" in result:
        return None
    if "NO MULTI-HOP VISUAL CONTRADICTION FOUND" in result:
        return None
    if "CONTRADICTION FOUND" not in result:
        return None

    parsed = {"raw_analysis": result, "is_visual": True}

    # Type
    m = re.search(r'Type:\s*(Numerical|Trend|Scope|Categorical|Temporal|Omission)', result, re.I)
    parsed["type"] = m.group(1) if m else "Numerical"

    # Chart Reference
    m = re.search(r'Chart Reference:\s*(.+?)(?:\n|$)', result)
    parsed["chart_ref"] = m.group(1).strip() if m else ""

    # Text Section
    m = re.search(r'Text Section:\s*(.+?)(?:\n|$)', result)
    parsed["text_section"] = m.group(1).strip() if m else ""

    # Chart Claim
    m = re.search(r'Chart Claim:\s*(.+?)(?:\n|$)', result)
    parsed["chart_claim"] = m.group(1).strip() if m else ""

    # Text Claim
    m = re.search(r'Text Claim:\s*(.+?)(?:\n|$)', result)
    parsed["text_claim"] = m.group(1).strip() if m else ""

    # Hop Chain (for multi-hop)
    m = re.search(r'Hop Chain:\s*(.+?)(?:\n|$)', result)
    parsed["hop_chain"] = m.group(1).strip() if m else ""

    # Explanation
    m = re.search(r'Explanation:\s*(.+?)(?:\n(?:Severity|Confidence)|$)', result, re.S)
    parsed["explanation"] = m.group(1).strip() if m else ""

    # Severity
    m = re.search(r'Severity:\s*(High|Medium|Low)', result, re.I)
    parsed["severity"] = m.group(1).capitalize() if m else "Low"

    # Confidence
    m = re.search(r'Confidence:\s*(\d+)', result)
    parsed["confidence"] = min(100, max(0, int(m.group(1)))) if m else 60

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# CHECKER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")
    return Groq(api_key=api_key)


def check_visual_textual_contradiction(
    chart_chunk: dict,
    text_chunks: list[dict],
    retries: int = 3,
) -> str:
    """
    Check if a chart's data contradicts the related text sections.

    Args:
        chart_chunk:  dict with 'text' (the chart description) and 'label'
        text_chunks:  list of related text chunk dicts
        retries:      number of retry attempts on rate limit

    Returns:
        raw LLM response string
    """
    text_context = ""
    for tc in text_chunks:
        text_context += f"\n\n[{tc['label']}]:\n{tc['text']}"

    user_message = (
        f"CHART DESCRIPTION [{chart_chunk['label']}]:\n"
        f"{chart_chunk['text']}\n\n"
        f"TEXT SECTIONS FROM THE SAME DOCUMENT:{text_context}\n\n"
        f"Does the chart data contradict any claim in the text sections above? "
        f"Be precise — only flag genuine inconsistencies."
    )

    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": VISUAL_TEXTUAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(2 ** attempt * 10)
            else:
                return "NO VISUAL-TEXTUAL CONTRADICTION FOUND"

    return "NO VISUAL-TEXTUAL CONTRADICTION FOUND"


def check_multi_hop_visual(
    chart_chunk: dict,
    primary_text: dict,
    secondary_texts: list[dict],
    retries: int = 3,
) -> str:
    """
    Multi-hop check: chart + primary text + secondary texts.
    Detects contradictions that only emerge when combining visual
    and multiple textual sources.
    """
    secondary_context = ""
    for st in secondary_texts:
        secondary_context += f"\n\n[{st['label']}]:\n{st['text']}"

    user_message = (
        f"CHART DESCRIPTION [{chart_chunk['label']}]:\n"
        f"{chart_chunk['text']}\n\n"
        f"PRIMARY TEXT SECTION [{primary_text['label']}]:\n"
        f"{primary_text['text']}\n\n"
        f"SECONDARY TEXT SECTIONS:{secondary_context}\n\n"
        f"Is there a multi-hop contradiction where the combination of the "
        f"chart data and one text section conflicts with another text section?"
    )

    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": MULTI_HOP_VISUAL_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(2 ** attempt * 10)
            else:
                return "NO MULTI-HOP VISUAL CONTRADICTION FOUND"

    return "NO MULTI-HOP VISUAL CONTRADICTION FOUND"


# ─────────────────────────────────────────────────────────────────────────────
# FULL VISUAL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_visual_contradiction_pipeline(
    chart_chunks: list[dict],
    text_chunks:  list[dict],
    retriever,
    top_k: int = 5,
    delay: float = 2.0,
    min_confidence: int = 60,
    enable_multi_hop: bool = True,
    progress_callback=None,
) -> tuple[list[dict], int]:
    """
    Run visual-textual contradiction detection across all chart chunks.

    For each chart:
      1. Retrieve top-K most relevant text sections (hybrid search)
      2. Check for direct visual-textual contradictions
      3. Optionally check for multi-hop contradictions

    Args:
        chart_chunks:     list of chart chunk dicts (from charts_to_chunks)
        text_chunks:      list of text chunk dicts
        retriever:        HybridRetriever instance (already built)
        top_k:            number of text sections to compare each chart against
        delay:            seconds between API calls
        min_confidence:   minimum confidence to keep a contradiction
        enable_multi_hop: whether to run the more expensive multi-hop check
        progress_callback: fn(current, total, label)

    Returns:
        (contradictions_list, clean_count)
    """
    contradictions = []
    clean_count = 0
    total = len(chart_chunks)

    for i, chart in enumerate(chart_chunks):
        if progress_callback:
            progress_callback(i, total, chart["label"])

        # Retrieve related text sections using hybrid search
        related_text = retriever.search(chart["text"], k=top_k)

        # Filter out other chart chunks — we only want text vs chart
        related_text = [r for r in related_text if not r.get("is_chart", False)]

        if not related_text:
            clean_count += 1
            continue

        if i > 0:
            time.sleep(delay)

        # ── Direct visual-textual check ──────────────────────────────────
        result = check_visual_textual_contradiction(chart, related_text)
        parsed = parse_visual_contradiction(result)

        if parsed and parsed.get("confidence", 0) >= min_confidence:
            contradictions.append({
                "target":           chart["label"],
                "target_display":   chart.get("display", chart["label"]),
                "target_text":      chart["text"],
                "analysis":         result,
                "related_sections": [r["label"] for r in related_text],
                "severity":         parsed["severity"],
                "confidence":       parsed["confidence"],
                "type":             f"Visual-{parsed['type']}",
                "is_visual":        True,
                "chart_claim":      parsed.get("chart_claim", ""),
                "text_claim":       parsed.get("text_claim", ""),
                "chart_data":       chart.get("chart_data", {}),
                "page_number":      chart.get("page_number", 0),
            })
        else:
            clean_count += 1

        # ── Multi-hop visual check ───────────────────────────────────────
        if enable_multi_hop and related_text:
            time.sleep(delay)

            primary = related_text[0]
            secondary = related_text[1:] if len(related_text) > 1 else []

            if secondary:
                mh_result = check_multi_hop_visual(chart, primary, secondary)
                mh_parsed = parse_visual_contradiction(mh_result)

                if mh_parsed and mh_parsed.get("confidence", 0) >= min_confidence:
                    contradictions.append({
                        "target":           chart["label"],
                        "target_display":   f"[Multi-hop] {chart.get('display', chart['label'])}",
                        "target_text":      chart["text"],
                        "analysis":         mh_result,
                        "related_sections": [primary["label"]] + [s["label"] for s in secondary],
                        "severity":         mh_parsed["severity"],
                        "confidence":       mh_parsed["confidence"],
                        "type":             f"Multi-hop-Visual-{mh_parsed.get('type', 'Unknown')}",
                        "is_visual":        True,
                        "is_multi_hop":     True,
                        "hop_chain":        mh_parsed.get("hop_chain", ""),
                        "chart_data":       chart.get("chart_data", {}),
                        "page_number":      chart.get("page_number", 0),
                    })

    return contradictions, clean_count
