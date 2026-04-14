import sys
import os
import time
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

import numpy as np
import streamlit as st

from pipeline.document_processor import extract_text_from_pdf, semantic_chunk, label_chunks
from pipeline.embedder import embed_chunks, build_faiss_index
from pipeline.hybrid_retriever import HybridRetriever
from pipeline.contradiction_checker import (
    run_full_pipeline, auto_detect_params, verify_contradictions_with_llm,
)
from pipeline.numeric_extractor import find_numeric_contradictions
from pipeline.chroma_store import store_chunks, load_chunks, list_stored_documents, delete_document
from pipeline.qa_chat import answer_question
from pipeline.visualiser import (
    make_severity_chart, make_type_pie,
    make_confidence_histogram, make_heatmap,
    make_network_graph, make_section_risk_bar,
    make_visual_contradiction_summary, make_cross_modal_network,
)
from pipeline.report_generator import generate_pdf_report
from pipeline.chart_analyzer import ChartAnalyzer
from pipeline.visual_contradiction_checker import run_visual_contradiction_pipeline
from pipeline.graph_store import (
    is_available as neo4j_available,
    build_graph, get_graph_stats,
    get_transitive_contradictions,
    get_entity_contradiction_clusters,
    get_most_contradictory_sections,
    get_graph,
    NEO4J_AVAILABLE,
)
from graph_tab_integration import render_graph_tab

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContradictAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "chunks":                   None,
    "embeddings":               None,
    "index":                    None,
    "retriever":                None,
    "contradictions":           None,
    "clean_count":              None,
    "heatmap_data":             None,
    "doc_name":                 None,
    "analysis_done":            False,
    "chat_history":             [],
    "pdf_bytes":                None,
    "auto_params":              None,
    "raw_text":                 None,
    "params_detected":          False,
    # ── Chart / Visual Analysis ──
    "chart_chunks":             None,   # chart-derived text chunks
    "chart_data_list":          None,   # raw ChartData objects (as dicts)
    "visual_contradictions":    None,   # visual-textual contradiction results
    "visual_clean_count":       None,
    "chart_analysis_done":      False,
    "enable_chart_analysis":    True,   # user toggle
    "file_bytes":               None,   # keep raw PDF for chart extraction
    # ── Knowledge Graph (Neo4j) ──
    "graph_built":              False,
    "graph_stats":              None,
    "transitive_contradictions": None,
    "entity_clusters":          None,
    "section_centrality":       None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #070710;
    color: #ddddf5;
}
.stApp { background-color: #070710; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #070710; }
::-webkit-scrollbar-thumb { background: #4040ff; border-radius: 2px; }

/* ── Noise texture overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Hero ── */
.hero-wrap {
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.35em;
    color: #4040ff;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 5.5rem;
    font-weight: 800;
    line-height: 0.9;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin: 0 0 1.2rem;
}
.hero-title .accent { color: #4040ff; }
.hero-title .dim { color: #2a2a5a; }
.hero-desc {
    font-size: 1rem;
    color: #6868a0;
    font-weight: 400;
    max-width: 520px;
    line-height: 1.6;
}
.hero-line {
    position: absolute;
    right: 0; top: 50%;
    width: 40%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #4040ff33, #4040ff88, #4040ff33, transparent);
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, #4040ff44, transparent);
    margin: 1.5rem 0 2.5rem;
    border: none;
}

/* ── Auto-param pill ── */
.param-panel {
    background: #0e0e1f;
    border: 1px solid #1c1c3a;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.param-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, #4040ff, #8888ff, #4040ff);
}
.param-panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: #4040ff;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.param-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.param-pill {
    background: #14142a;
    border: 1px solid #2020408a;
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.param-pill-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #5555aa;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.param-pill-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #aaaaff;
    font-weight: 500;
}
.param-reasoning {
    font-size: 0.8rem;
    color: #5555aa;
    margin-top: 0.9rem;
    font-style: italic;
    border-left: 2px solid #2020488a;
    padding-left: 0.8rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.8rem;
}
.metric-card {
    flex: 1;
    background: #0e0e1f;
    border: 1px solid #1c1c3a;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: var(--accent-color, #4040ff44);
}
.metric-card.red   { --accent-color: #ff404066; }
.metric-card.amber { --accent-color: #ff990066; }
.metric-card.green { --accent-color: #00dd8866; }
.metric-card.blue  { --accent-color: #4040ff66; }
.m-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    color: #5050a0;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.m-value {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -0.02em;
}
.m-sub {
    font-size: 0.72rem;
    color: #5050a0;
    margin-top: 0.3rem;
}

/* ── Status bar ── */
.status-bar {
    background: #0e0e1f;
    border: 1px solid #1c1c3a;
    border-radius: 10px;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4040ff;
    flex-shrink: 0;
    box-shadow: 0 0 8px #4040ff;
    animation: glow 2s infinite;
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 6px #4040ff; }
    50%       { box-shadow: 0 0 14px #6666ff; }
}
.status-text { font-size: 0.82rem; color: #7070b0; }

/* ── Contradiction card ── */
.c-card {
    background: #0b0b1a;
    border: 1px solid #181830;
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s, transform 0.15s;
    position: relative;
}
.c-card:hover {
    border-color: #4040ff44;
    transform: translateY(-1px);
}
.c-card-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.c-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #4040aa;
    letter-spacing: 0.08em;
    min-width: 2rem;
}
.c-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #ddddf5;
    flex: 1;
}
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.08em;
    padding: 3px 9px;
    border-radius: 5px;
    text-transform: uppercase;
    font-weight: 600;
    white-space: nowrap;
}
.b-high   { background: #ff404018; color: #ff6060; border: 1px solid #ff404035; }
.b-medium { background: #ff990018; color: #ffaa44; border: 1px solid #ff990035; }
.b-low    { background: #ffdd0018; color: #ffdd55; border: 1px solid #ffdd0035; }
.b-type   { background: #4040ff18; color: #8888ff; border: 1px solid #4040ff35; }
.b-conf   { background: #00dd8818; color: #00dd88; border: 1px solid #00dd8835; }

.c-text {
    background: #070710;
    border: 1px solid #181830;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7070a8;
    line-height: 1.65;
    margin: 0.8rem 0;
}
.c-analysis {
    font-size: 0.88rem;
    color: #b0b0d8;
    line-height: 1.75;
    margin: 0.8rem 0;
}
.c-sources {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #3535aa;
    letter-spacing: 0.06em;
    margin-top: 0.8rem;
}

/* ── Upload zone ── */
.upload-card {
    background: #0e0e1f;
    border: 1px dashed #252548;
    border-radius: 20px;
    padding: 3.5rem 2rem;
    text-align: center;
    transition: border-color 0.25s, background 0.25s;
    margin-bottom: 1.5rem;
    cursor: pointer;
}
.upload-card:hover {
    border-color: #4040ff;
    background: #0f0f25;
}
.upload-icon { font-size: 3rem; margin-bottom: 0.8rem; display: block; }
.upload-h { font-size: 1.1rem; font-weight: 700; color: #ddddf5; }
.upload-s { font-size: 0.82rem; color: #5050a0; margin-top: 0.4rem; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: #5050a0;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; display: block; }
.empty-h { font-size: 1.1rem; font-weight: 700; color: #8080c0; }
.empty-s { font-size: 0.82rem; margin-top: 0.4rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #08081a !important;
    border-right: 1px solid #141430 !important;
}
.sb-head {
    padding: 1.5rem 0 1rem;
}
.sb-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.sb-sub {
    font-size: 0.72rem;
    color: #5050a0;
    margin-top: 0.2rem;
}
.sb-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    color: #4040ff;
    text-transform: uppercase;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #141430;
}
.sb-param-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid #0f0f28;
}
.sb-param-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #5050a0;
}
.sb-param-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #aaaaff;
    font-weight: 500;
}
.sb-mode-auto {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #4040ff18;
    border: 1px solid #4040ff44;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #8888ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.sb-mode-manual {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #ff990018;
    border: 1px solid #ff990044;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #ffaa44;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Buttons ── */
.stButton > button {
    background: #4040ff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #5555ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px #4040ff44 !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #0e0e1f !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid #1c1c3a !important;
    gap: 2px !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #5050a0 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1a1a38 !important;
    color: #ddddf5 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0e0e1f !important;
    border: 1px solid #1c1c3a !important;
    border-radius: 10px !important;
}

/* ── Inputs ── */
.stTextInput > div > input,
[data-testid="stSelectbox"] > div {
    background: #0e0e1f !important;
    border-color: #1c1c3a !important;
    color: #ddddf5 !important;
}

/* ── Progress ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4040ff, #8080ff) !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: #0e0e1f !important;
    border: 1px solid #1c1c3a !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stChatInputTextArea"] {
    background: #0e0e1f !important;
    border-color: #1c1c3a !important;
    color: #ddddf5 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0e0e1f !important;
    border: 1px dashed #252548 !important;
    border-radius: 12px !important;
}

/* ── Section tags ── */
.sec-tag {
    display: inline-block;
    background: #14142a;
    border-radius: 5px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #6666cc;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-head">
        <div class="sb-title">ContradictAI</div>
        <div class="sb-sub">Hybrid RAG · LLaMA 3.3 70B · Groq</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameter mode toggle ─────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Parameters</div>', unsafe_allow_html=True)
    param_mode = st.radio(
        "Mode",
        ["🤖 Auto (LLM decides)", "🔧 Manual override"],
        label_visibility="collapsed"
    )
    auto_mode = param_mode.startswith("🤖")

    if auto_mode:
        st.markdown('<div class="sb-mode-auto">⚡ AI auto-config active</div>',
                    unsafe_allow_html=True)
        if st.session_state.auto_params:
            p = st.session_state.auto_params
            st.markdown(f"""
            <div style="margin-top:0.8rem;">
                <div class="sb-param-row"><span class="sb-param-key">top_k</span><span class="sb-param-val">{p['top_k']}</span></div>
                <div class="sb-param-row"><span class="sb-param-key">alpha</span><span class="sb-param-val">{p['alpha']}</span></div>
                <div class="sb-param-row"><span class="sb-param-key">min_chunk</span><span class="sb-param-val">{p['min_chunk']}</span></div>
                <div class="sb-param-row"><span class="sb-param-key">max_chunk</span><span class="sb-param-val">{p['max_chunk']}</span></div>
                <div class="sb-param-row"><span class="sb-param-key">api_delay</span><span class="sb-param-val">2.0s</span></div>
                <div class="sb-param-row"><span class="sb-param-key">min_conf</span><span class="sb-param-val">{p['min_confidence']}%</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("Upload a doc to auto-configure.")
        # Fixed values for auto mode
        top_k     = st.session_state.auto_params["top_k"]          if st.session_state.auto_params else 5
        alpha     = st.session_state.auto_params["alpha"]           if st.session_state.auto_params else 0.5
        min_chunk = st.session_state.auto_params["min_chunk"]       if st.session_state.auto_params else 150
        max_chunk = st.session_state.auto_params["max_chunk"]       if st.session_state.auto_params else 1000
        api_delay = 2.0
        min_conf  = st.session_state.auto_params["min_confidence"]  if st.session_state.auto_params else 70
    else:
        st.markdown('<div class="sb-mode-manual">⚙ manual override</div>',
                    unsafe_allow_html=True)
        top_k     = st.slider("Top-K sections",    2,    8,    5)
        alpha     = st.slider("Dense weight α",    0.0,  1.0,  0.5, 0.1,
                              help="0 = pure BM25 · 1 = pure FAISS")
        min_chunk = st.slider("Min chunk (chars)", 50,   300,  150)
        max_chunk = st.slider("Max chunk (chars)", 300,  2000, 1000)
        api_delay = 2.0  # Always 2.0 for Groq free tier
        min_conf  = st.slider("Min confidence %",  60,   95,   70)
        st.caption("⚡ API delay fixed at 2s for Groq free tier.")

    # ── Chart Analysis Toggle ────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Multimodal</div>', unsafe_allow_html=True)
    st.session_state.enable_chart_analysis = st.checkbox(
        "📊 Chart / Graph Analysis",
        value=st.session_state.enable_chart_analysis,
        help="Extract charts from PDF and detect visual-textual contradictions"
    )
    if st.session_state.enable_chart_analysis:
        st.caption("Vision model will analyse charts in your PDF.")
    else:
        st.caption("Text-only mode — charts will be ignored.")

    # ── Stored docs ───────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Stored Docs</div>', unsafe_allow_html=True)
    stored = list_stored_documents()
    if stored:
        for d in stored:
            c1, c2 = st.columns([4, 1])
            c1.caption(d[:22] + "…" if len(d) > 22 else d)
            if c2.button("✕", key=f"del_{d}"):
                delete_document(d)
                st.rerun()
    else:
        st.caption("No cached documents.")

    # ── Session ───────────────────────────────────────────────────────────────
    if st.session_state.analysis_done:
        st.markdown('<div class="sb-section">Session</div>', unsafe_allow_html=True)
        if st.button("🗑 Clear Results"):
            for k in ["chunks", "embeddings", "index", "retriever", "contradictions",
                      "clean_count", "heatmap_data", "doc_name", "analysis_done",
                      "pdf_bytes", "auto_params", "raw_text", "params_detected",
                      "chart_chunks", "chart_data_list", "visual_contradictions",
                      "visual_clean_count", "chart_analysis_done", "file_bytes"]:
                if k in ("analysis_done", "params_detected", "chart_analysis_done"):
                    st.session_state[k] = False
                else:
                    st.session_state[k] = None
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
col_h, col_spacer = st.columns([3, 1])
with col_h:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">⚡ Generative AI · Hybrid RAG · Document Intelligence</div>
        <h1 class="hero-title">Contra<span class="accent">dict</span><span class="dim">AI</span></h1>
        <p class="hero-desc">
            Surface logical contradictions in company reports — powered by hybrid retrieval
            and LLaMA 3.3 70B reasoning.
        </p>
        <div class="hero-line"></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_detect, tab_chat, tab_graph = st.tabs([
    "⚡  Contradiction Detection",
    "💬  Document Q&A",
    "🕸️  Graph Intelligence",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:

    # ── Upload ────────────────────────────────────────────────────────────────
    if not st.session_state.analysis_done:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown("""
            <div class="upload-card">
                <span class="upload-icon">📄</span>
                <div class="upload-h">Drop your company report here</div>
                <div class="upload-s">Annual reports · Financial filings · Policy docs · Compliance</div>
            </div>
            """, unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload PDF", type=["pdf"],
                label_visibility="collapsed",
                help="PDF up to ~50 pages"
            )
    else:
        uploaded = None

    # ── Process uploaded file ─────────────────────────────────────────────────
    if uploaded is not None and not st.session_state.analysis_done:
        doc_name = uploaded.name
        st.session_state.doc_name = doc_name
        file_bytes = uploaded.read()
        st.session_state.file_bytes = file_bytes    # keep for chart extraction

        cached = load_chunks(doc_name)

        if cached:
            st.session_state.chunks = cached
            # Auto-detect params from cached chunks' text if in auto mode
            if auto_mode and not st.session_state.auto_params:
                with st.spinner("🤖 LLM is calibrating parameters for this document…"):
                    combined_text = " ".join(c["text"] for c in cached[:5])
                    st.session_state.auto_params = auto_detect_params(combined_text)
                    st.session_state.params_detected = True
                    # Update variables from auto params
                    top_k     = st.session_state.auto_params["top_k"]
                    alpha     = st.session_state.auto_params["alpha"]
                    min_chunk = st.session_state.auto_params["min_chunk"]
                    max_chunk = st.session_state.auto_params["max_chunk"]
                    min_conf  = st.session_state.auto_params["min_confidence"]

            with st.spinner("⚡ Rebuilding index from cache…"):
                emb = embed_chunks(cached)
                idx = build_faiss_index(emb)
                st.session_state.embeddings = emb
                st.session_state.index      = idx
                st.session_state.retriever  = HybridRetriever(cached, emb, idx)
            st.success(f"📁 Loaded **{len(cached)} sections** from cache")
        else:
            # Extract text
            with st.spinner("📄 Extracting text…"):
                raw = extract_text_from_pdf(file_bytes)
            if not raw.strip():
                st.error("No text found — PDF may be scanned/image-based.")
                st.stop()

            st.session_state.raw_text = raw

            # Auto-detect params BEFORE chunking (so chunk sizes are correct)
            if auto_mode:
                with st.spinner("🤖 LLM is calibrating parameters for this document…"):
                    st.session_state.auto_params = auto_detect_params(raw)
                    st.session_state.params_detected = True
                    top_k     = st.session_state.auto_params["top_k"]
                    alpha     = st.session_state.auto_params["alpha"]
                    min_chunk = st.session_state.auto_params["min_chunk"]
                    max_chunk = st.session_state.auto_params["max_chunk"]
                    min_conf  = st.session_state.auto_params["min_confidence"]

            with st.spinner("🔪 Chunking…"):
                chunks = label_chunks(semantic_chunk(raw, min_len=min_chunk, max_len=max_chunk))
            if not chunks:
                st.error("No sections found — try adjusting chunk parameters.")
                st.stop()

            with st.spinner("🧠 Embedding…"):
                emb = embed_chunks(chunks)
                idx = build_faiss_index(emb)

            store_chunks(doc_name, chunks, emb)
            st.session_state.chunks     = chunks
            st.session_state.embeddings = emb
            st.session_state.index      = idx
            st.session_state.retriever  = HybridRetriever(chunks, emb, idx)

        chunks = st.session_state.chunks
        # Estimate: NLI scan (~15s) + 3 optional LLM enrichment calls
        est = round((15 + 3 * (api_delay + 5.0)) / 60, 1)

        # ── Show auto-params panel if in auto mode ─────────────────────────
        if auto_mode and st.session_state.auto_params:
            p = st.session_state.auto_params
            st.markdown(f"""
            <div class="param-panel">
                <div class="param-panel-label">⚡ LLM-calibrated parameters</div>
                <div class="param-grid">
                    <div class="param-pill">
                        <span class="param-pill-key">top_k</span>
                        <span class="param-pill-val">{p['top_k']}</span>
                    </div>
                    <div class="param-pill">
                        <span class="param-pill-key">alpha</span>
                        <span class="param-pill-val">{p['alpha']}</span>
                    </div>
                    <div class="param-pill">
                        <span class="param-pill-key">min_chunk</span>
                        <span class="param-pill-val">{p['min_chunk']} chars</span>
                    </div>
                    <div class="param-pill">
                        <span class="param-pill-key">max_chunk</span>
                        <span class="param-pill-val">{p['max_chunk']} chars</span>
                    </div>
                    <div class="param-pill">
                        <span class="param-pill-key">api_delay</span>
                        <span class="param-pill-val">2.0s</span>
                    </div>
                    <div class="param-pill">
                        <span class="param-pill-key">min_conf</span>
                        <span class="param-pill-val">{p['min_confidence']}%</span>
                    </div>
                </div>
                <div class="param-reasoning">💡 {p.get('reasoning', '')}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Pre-run stats ──────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card blue">
                <div class="m-label">Sections</div>
                <div class="m-value">{len(chunks)}</div>
                <div class="m-sub">semantic chunks</div>
            </div>
            <div class="metric-card blue">
                <div class="m-label">Avg Length</div>
                <div class="m-value">{int(np.mean([len(c['text']) for c in chunks]))}</div>
                <div class="m-sub">characters</div>
            </div>
            <div class="metric-card amber">
                <div class="m-label">Est. Time</div>
                <div class="m-value">{est}</div>
                <div class="m-sub">minutes</div>
            </div>
            <div class="metric-card blue">
                <div class="m-label">Max Pairs</div>
                <div class="m-value">{len(chunks) * top_k}</div>
                <div class="m-sub">comparisons</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _, bcol, _ = st.columns([1, 1, 1])
        with bcol:
            if st.button("⚡ Run Contradiction Detection"):
                st.markdown("""
                <div class="status-bar">
                    <div class="status-dot"></div>
                    <div class="status-text">Hybrid RAG pipeline running — do not close this tab</div>
                </div>
                """, unsafe_allow_html=True)

                import math as _math

                phase_label = st.empty()
                prog = st.progress(0)
                status = st.empty()
                timer_display = st.empty()

                _phase_names = {
                    "nli_scan": "Stage 1: NLI pre-filter (local, fast)",
                    "build_results": "Building candidate list",
                    "llm_verify": "Stage 2: Groq verification (filtering false positives)",
                    "llm_enrich": "Enriching explanations",
                    "analyze": "Analyzing sections",
                }

                def on_progress(cur, total, phase, label, t0):
                    pct = min(int((cur + 1) / total * 100), 99)
                    prog.progress(pct)

                    elapsed = time.time() - t0
                    if cur > 0:
                        per_step = elapsed / cur
                        remaining = per_step * (total - cur)
                        eta_str = f"{int(remaining)}s left" if remaining < 120 else f"{remaining/60:.1f}min left"
                    else:
                        eta_str = "estimating…"

                    phase_name = _phase_names.get(phase, phase)
                    phase_label.markdown(
                        f"<div style='font-size:0.78rem;color:#58a6ff;font-weight:600;"
                        f"font-family:JetBrains Mono,monospace;'>"
                        f"⏳ {phase_name}</div>",
                        unsafe_allow_html=True
                    )
                    status.caption(f"→ {label}  ·  step {cur+1}/{total}  ·  {pct}%")

                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    timer_display.caption(f"⏱ {mins:02d}:{secs:02d} elapsed  ·  ~{eta_str}")

                c, cl, hm = run_full_pipeline(
                    chunks=chunks,
                    embeddings=st.session_state.embeddings,
                    index=st.session_state.index,
                    top_k=top_k,
                    delay_between_calls=api_delay,
                    min_confidence=min_conf,
                    alpha=alpha,
                    progress_callback=on_progress
                )
                prog.progress(100)
                phase_label.empty()
                status.empty()
                timer_display.empty()

                # ── NUMERIC CONTRADICTION PASS (deterministic, ~100ms) ──
                # Regex-based (entity, metric, value) extraction. Adds to
                # the contradiction list but never removes anything. Fail-
                # open: any exception is swallowed inside the helper.
                try:
                    numeric_c = find_numeric_contradictions(chunks) or []
                    if numeric_c:
                        # Dedup against existing NLI results by (target, first related)
                        existing_keys = {
                            (x.get("target", ""),
                             (x.get("related_sections") or [""])[0]
                                 .replace(" (same section)", ""))
                            for x in c
                        }
                        for nc in numeric_c:
                            key = (nc.get("target", ""),
                                   (nc.get("related_sections") or [""])[0])
                            if key in existing_keys:
                                continue
                            c.append(nc)
                except Exception as _e:
                    logger.warning(f"numeric extractor skipped: {_e}")

                # ── LLM VERIFIER (parallel, fail-open) ──────────────────
                # Second-pass check on uncertain NLI pairs (conf 60-85).
                # NUMERIC pairs and conf>=86 pairs are already trusted and
                # bypass this step. Runs 4-wide in parallel so the wall-
                # clock cost on a typical doc is ~1-2s.
                try:
                    phase_label.markdown(
                        "<div style='font-size:0.78rem;color:#ffb347;font-weight:600;"
                        "font-family:JetBrains Mono,monospace;'>"
                        "🔍 LLM Verification (parallel)</div>",
                        unsafe_allow_html=True
                    )
                    c = verify_contradictions_with_llm(c, parallel=4)
                except Exception as _e:
                    logger.warning(f"LLM verifier skipped: {_e}")
                finally:
                    phase_label.empty()

                st.session_state.contradictions = c
                st.session_state.clean_count    = cl
                st.session_state.heatmap_data   = hm
                st.session_state.analysis_done  = True

                # ── CHART ANALYSIS (Phase 2 — Visual-Textual) ────────────
                if st.session_state.enable_chart_analysis and st.session_state.file_bytes:
                    prog.progress(0)
                    _chart_t0 = time.time()

                    def on_chart_progress(cur, total, msg):
                        if total > 0:
                            pct = min(int((cur + 1) / total * 100), 99)
                            prog.progress(pct)
                        elapsed = time.time() - _chart_t0
                        mins = int(elapsed) // 60
                        secs = int(elapsed) % 60
                        phase_label.markdown(
                            "<div style='font-size:0.78rem;color:#d29922;font-weight:600;"
                            "font-family:JetBrains Mono,monospace;'>"
                            "📊 Chart & Graph Analysis</div>",
                            unsafe_allow_html=True
                        )
                        status.caption(f"→ {msg}")
                        timer_display.caption(f"⏱ {mins:02d}:{secs:02d} elapsed")

                    try:
                        analyzer = ChartAnalyzer()

                        chart_data_list = analyzer.analyze_pdf(
                            st.session_state.file_bytes,
                            progress_callback=on_chart_progress,
                        )

                        if chart_data_list:
                            chart_chunks = analyzer.charts_to_chunks(chart_data_list)
                            st.session_state.chart_chunks    = chart_chunks
                            st.session_state.chart_data_list = [cd.to_dict() for cd in chart_data_list]

                            # Run visual-textual contradiction detection
                            phase_label.markdown(
                                "<div style='font-size:0.78rem;color:#8844ff;font-weight:600;"
                                "font-family:JetBrains Mono,monospace;'>"
                                "🔬 Visual-Textual Cross-referencing</div>",
                                unsafe_allow_html=True
                            )
                            prog.progress(0)
                            vc, vcl = run_visual_contradiction_pipeline(
                                chart_chunks=chart_chunks,
                                text_chunks=chunks,
                                retriever=st.session_state.retriever,
                                top_k=top_k,
                                delay=api_delay,
                                min_confidence=min_conf,
                                enable_multi_hop=True,
                                progress_callback=on_chart_progress,
                            )
                            st.session_state.visual_contradictions = vc
                            st.session_state.visual_clean_count    = vcl
                            st.session_state.chart_analysis_done   = True

                            # Merge visual contradictions into main list
                            st.session_state.contradictions = c + vc

                        else:
                            st.session_state.chart_analysis_done = True
                            st.session_state.chart_chunks = []
                            st.session_state.visual_contradictions = []
                            st.session_state.visual_clean_count = 0

                    except Exception as e:
                        st.session_state.chart_analysis_done = False
                        status.caption(f"⚠️ Chart analysis error: {str(e)[:100]}")

                prog.progress(100)
                phase_label.empty()
                status.empty()
                timer_display.empty()

                try:
                    all_c = st.session_state.contradictions
                    st.session_state.pdf_bytes = generate_pdf_report(
                        doc_name, doc_name, len(chunks), all_c, cl)
                except Exception:
                    st.session_state.pdf_bytes = None

                # ── NEO4J KNOWLEDGE GRAPH (optional) ────────────────────
                try:
                    if neo4j_available():
                        phase_label.markdown(
                            "<div style='font-size:0.78rem;color:#00dd88;font-weight:600;"
                            "font-family:JetBrains Mono,monospace;'>"
                            "🕸 Building Knowledge Graph</div>",
                            unsafe_allow_html=True
                        )
                        all_c = st.session_state.contradictions or []
                        build_graph(doc_name, chunks, all_c)
                        st.session_state.graph_stats = get_graph_stats(doc_name)
                        st.session_state.transitive_contradictions = get_transitive_contradictions(doc_name)
                        st.session_state.entity_clusters = get_entity_contradiction_clusters(doc_name)
                        st.session_state.section_centrality = get_most_contradictory_sections(doc_name)
                        st.session_state.graph_built = True
                        phase_label.empty()
                except Exception as e:
                    st.session_state.graph_built = False
                    logger.warning(f"Neo4j graph build skipped: {e}")

                st.rerun()

    # ── RESULTS ──────────────────────────────────────────────────────────────
    if st.session_state.analysis_done:
        C      = st.session_state.contradictions or []
        cl     = st.session_state.clean_count
        chunks = st.session_state.chunks or []
        hm     = st.session_state.heatmap_data or {}

        # Separate text-only vs visual contradictions
        text_only_C  = [c for c in C if not c.get("is_visual")]
        visual_C     = [c for c in C if c.get("is_visual")]

        high   = sum(1 for c in C if c.get("severity") == "High")
        medium = sum(1 for c in C if c.get("severity") == "Medium")
        low    = sum(1 for c in C if c.get("severity") == "Low")
        n_visual = len(visual_C)
        n_charts = len(st.session_state.chart_chunks or [])

        # Param summary strip (auto mode)
        if auto_mode and st.session_state.auto_params:
            p = st.session_state.auto_params
            st.markdown(f"""
            <div class="status-bar" style="margin-top:1rem;">
                <div class="status-dot" style="background:#00dd88;box-shadow:0 0 8px #00dd88;animation:none;"></div>
                <div class="status-text">
                    Auto-configured · top_k={p['top_k']} · α={p['alpha']} ·
                    min_conf={p['min_confidence']}% · {p.get('reasoning','')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row" style="margin-top:1rem;">
            <div class="metric-card {'red' if len(C) > 0 else 'green'}">
                <div class="m-label">Contradictions</div>
                <div class="m-value">{len(C)}</div>
                <div class="m-sub">{len(text_only_C)} textual · {n_visual} visual</div>
            </div>
            <div class="metric-card red">
                <div class="m-label">High Severity</div>
                <div class="m-value">{high}</div>
                <div class="m-sub">critical</div>
            </div>
            <div class="metric-card amber">
                <div class="m-label">Med · Low</div>
                <div class="m-value">{medium} · {low}</div>
                <div class="m-sub">warnings</div>
            </div>
            <div class="metric-card green">
                <div class="m-label">Clean</div>
                <div class="m-value">{cl}</div>
                <div class="m-sub">consistent sections</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chart extraction info bar
        if st.session_state.chart_analysis_done and n_charts > 0:
            st.markdown(f"""
            <div class="status-bar" style="margin-top:0.5rem;">
                <div class="status-dot" style="background:#8888ff;box-shadow:0 0 8px #8888ff;animation:none;"></div>
                <div class="status-text">
                    📊 Multimodal analysis · {n_charts} chart(s) extracted ·
                    {n_visual} visual-textual contradiction(s) found
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                    color:#5050a0;letter-spacing:0.1em;margin-bottom:1.5rem;'>
            ◈ {st.session_state.doc_name or 'document'}
             ·  {len(chunks)} sections
            {" · " + str(n_charts) + " charts" if n_charts else ""}
             ·  min confidence {min_conf}%
        </div>
        """, unsafe_allow_html=True)

        if not C:
            # ── CONSISTENCY REPORT (no contradictions) ──────────────────
            n_sect = len(chunks)
            n_pairs_checked = n_sect * (n_sect - 1) // 2
            st.markdown(f"""
            <div class="empty-state">
                <span class="empty-icon">✅</span>
                <div class="empty-h">Document Consistency Verified</div>
                <div class="empty-s">No logical contradictions detected above the confidence threshold.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Consistency Report")

            rep_cols = st.columns(4)
            rep_cols[0].metric("Sections Analyzed", n_sect)
            rep_cols[1].metric("Intra-section Checks", n_sect)
            rep_cols[2].metric("Cross-section Pairs", n_pairs_checked)
            rep_cols[3].metric("Confidence Threshold", f"{min_conf}%")

            st.markdown("##### Section-by-Section Status")
            for ch in chunks:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0.8rem;
                            margin:0.2rem 0;border-radius:0.4rem;background:#0d1117;border:1px solid #1a2332;">
                    <span style="color:#2ea043;font-size:1.1rem;">●</span>
                    <span style="color:#c9d1d9;font-size:0.82rem;font-weight:500;">{ch.get('display', ch['label'])}</span>
                    <span style="color:#3d4f5f;font-size:0.72rem;margin-left:auto;">{len(ch['text'])} chars · consistent</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("##### Analysis Summary")
            summary_text = (
                f"The document **{st.session_state.doc_name}** was analyzed using a hybrid RAG pipeline "
                f"(BM25 + FAISS semantic retrieval, α={alpha}) with LLaMA 3.3 70B as the reasoning model. "
                f"**{n_sect}** sections were identified and checked both internally (intra-section) "
                f"and against each other (cross-section, top-k={top_k}). "
                f"No contradictions met the minimum confidence threshold of **{min_conf}%**. "
                f"The document appears logically consistent."
            )
            st.markdown(summary_text)

            # Export for clean results
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            clean_report_lines = [
                "CONTRADICTAI — CONSISTENCY REPORT",
                "=" * 60,
                f"Document: {st.session_state.doc_name}",
                f"Sections: {n_sect}",
                f"Cross-section pairs checked: {n_pairs_checked}",
                f"Confidence threshold: {min_conf}%",
                f"Result: NO CONTRADICTIONS DETECTED",
                "=" * 60, "",
                "Section Status:",
            ]
            for ch in chunks:
                clean_report_lines.append(f"  [PASS] {ch.get('display', ch['label'])} ({len(ch['text'])} chars)")
            clean_report_lines += ["", "=" * 60,
                "Conclusion: The document appears logically consistent.",
                "All sections passed both intra-section and cross-section checks."]
            clean_txt = "\n".join(clean_report_lines)
            st.download_button("📄 Download Consistency Report", data=clean_txt,
                file_name=f"consistency_{st.session_state.doc_name}.txt",
                mime="text/plain", use_container_width=True)

        else:
            tab_labels = ["📋  Findings", "📊  Charts", "🗺️  Maps"]
            if visual_C:
                tab_labels.append("🔬  Visual-Textual")
            if st.session_state.graph_built:
                tab_labels.append("🕸  Knowledge Graph")
            result_tabs = st.tabs(tab_labels)
            r1, r2, r3 = result_tabs[0], result_tabs[1], result_tabs[2]

            # ── FINDINGS ─────────────────────────────────────────────────────
            with r1:
                f1, f2, f3 = st.columns(3)
                sev_f = f1.selectbox("Severity", ["All", "High", "Medium", "Low"], key="sf")
                all_types = ["All", "Direct", "Conditional", "Exception-based",
                             "Numerical", "Temporal", "Definitional", "Scope"]
                if visual_C:
                    all_types += ["Visual-Numerical", "Visual-Trend", "Visual-Scope",
                                  "Visual-Categorical", "Visual-Temporal", "Visual-Omission"]
                typ_f = f2.selectbox("Type", all_types, key="tf")
                mode_f = f3.selectbox("Source", ["All", "Text-only", "Visual"], key="mf")

                shown = 0
                for i, c in enumerate(C):
                    sev   = c.get("severity", "Low")
                    ctype = c.get("type", "Direct")
                    conf  = c.get("confidence", 0)
                    is_vis = c.get("is_visual", False)

                    if sev_f != "All" and sev != sev_f:     continue
                    if typ_f != "All" and ctype != typ_f:   continue
                    if mode_f == "Text-only" and is_vis:    continue
                    if mode_f == "Visual" and not is_vis:   continue

                    badge_sev = {"High": "b-high", "Medium": "b-medium", "Low": "b-low"}.get(sev, "b-low")
                    vis_badge = '<span class="badge" style="background:#8844ff18;color:#aa88ff;border:1px solid #8844ff35;">VISUAL</span>' if is_vis else ""

                    st.markdown(f"""
                    <div class="c-card">
                        <div class="c-card-header">
                            <div class="c-num">#{i+1:02d}</div>
                            <div class="c-title">{c.get('target_display', c['target'])}</div>
                            <span class="badge {badge_sev}">{sev}</span>
                            <span class="badge b-type">{ctype}</span>
                            <span class="badge b-conf">{conf}%</span>
                            {vis_badge}
                        </div>
                        <div class="c-text">{c['target_text'][:400].replace('<','&lt;').replace(chr(10),' ')}{'…' if len(c['target_text'])>400 else ''}</div>
                        <div class="c-analysis">{c['analysis'].replace(chr(10),'<br>').replace('<','&lt;').replace('&lt;br>','<br>')}</div>
                        <div class="c-sources">◈ compared against: {' · '.join(c['related_sections'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    shown += 1

                if shown == 0:
                    st.info("No results match the current filters.")

            # ── CHARTS ───────────────────────────────────────────────────────
            with r2:
                ca, cb = st.columns(2)
                with ca:
                    st.plotly_chart(make_severity_chart(C), use_container_width=True)
                with cb:
                    st.plotly_chart(make_type_pie(C), use_container_width=True)
                st.plotly_chart(make_confidence_histogram(C), use_container_width=True)
                st.plotly_chart(make_section_risk_bar(C, chunks), use_container_width=True)

            # ── MAPS ─────────────────────────────────────────────────────────
            with r3:
                st.markdown("#### Contradiction Heatmap")
                st.caption("Section pairs — darker = higher confidence contradiction")
                st.plotly_chart(make_heatmap(chunks, C), use_container_width=True)
                st.markdown("#### Contradiction Network")
                st.caption("Nodes = sections · Edge colour = severity · Blue = clean")
                st.plotly_chart(make_network_graph(chunks, C), use_container_width=True)

            # ── VISUAL-TEXTUAL TAB ──────────────────────────────────────────
            if visual_C and len(result_tabs) > 3:
                with result_tabs[3]:
                    st.markdown("#### Visual-Textual Contradictions")
                    st.caption(
                        "These contradictions were detected between **charts/graphs** "
                        "extracted from the PDF and **textual claims** in the document. "
                        "This cross-modal analysis is a novel capability."
                    )

                    # Summary metrics
                    vc1, vc2, vc3 = st.columns(3)
                    vc1.metric("Charts Extracted", n_charts)
                    vc2.metric("Visual Contradictions", n_visual)
                    multi_hop = sum(1 for v in visual_C if v.get("is_multi_hop"))
                    vc3.metric("Multi-hop", multi_hop)

                    st.divider()

                    # Visual contradiction type breakdown
                    if visual_C:
                        st.plotly_chart(
                            make_visual_contradiction_summary(visual_C),
                            use_container_width=True
                        )

                    st.divider()

                    # Detailed findings
                    for j, vc in enumerate(visual_C):
                        sev = vc.get("severity", "Low")
                        badge_s = {"High": "b-high", "Medium": "b-medium", "Low": "b-low"}.get(sev, "b-low")
                        mh_tag = " · 🔗 Multi-hop" if vc.get("is_multi_hop") else ""
                        page_tag = f" · p.{vc.get('page_number', '?')}" if vc.get("page_number") else ""

                        st.markdown(f"""
                        <div class="c-card" style="border-left: 3px solid #8844ff;">
                            <div class="c-card-header">
                                <div class="c-num">V{j+1:02d}</div>
                                <div class="c-title">{vc.get('target_display', vc['target'])}{page_tag}{mh_tag}</div>
                                <span class="badge {badge_s}">{sev}</span>
                                <span class="badge b-type">{vc.get('type', 'Visual')}</span>
                                <span class="badge b-conf">{vc.get('confidence', 0)}%</span>
                            </div>
                            <div class="c-text">{vc['target_text'][:400].replace('<','&lt;').replace(chr(10),' ')}{'…' if len(vc['target_text'])>400 else ''}</div>
                            <div class="c-analysis">{vc['analysis'].replace(chr(10),'<br>').replace('<','&lt;').replace('&lt;br>','<br>')}</div>
                            <div class="c-sources">◈ compared against: {' · '.join(vc.get('related_sections', []))}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Cross-modal network graph
                    if visual_C:
                        st.divider()
                        st.markdown("#### Cross-Modal Contradiction Network")
                        st.caption(
                            "Purple nodes = charts · Blue nodes = text sections · "
                            "Edges = detected contradictions"
                        )
                        st.plotly_chart(
                            make_cross_modal_network(
                                chunks, st.session_state.chart_chunks or [], visual_C
                            ),
                            use_container_width=True,
                        )

            # ── KNOWLEDGE GRAPH TAB ─────────────────────────────────────────
            if st.session_state.graph_built:
                kg_tab_idx = len(tab_labels) - 1
                with result_tabs[kg_tab_idx]:
                    st.markdown("#### Contradiction Knowledge Graph")
                    st.caption(
                        "Neo4j-powered graph analysis of contradiction relationships. "
                        "Shows transitive contradictions, entity clusters, and section centrality."
                    )

                    # Graph stats
                    gs = st.session_state.graph_stats or {}
                    if gs:
                        g1, g2, g3, g4, g5 = st.columns(5)
                        g1.metric("Sections", gs.get("sections", 0))
                        g2.metric("Statements", gs.get("statements", 0))
                        g3.metric("Entities", gs.get("entities", 0))
                        g4.metric("Contradiction Edges", gs.get("contradiction_edges", 0))
                        g5.metric("Mention Edges", gs.get("mention_edges", 0))

                    st.divider()

                    # Section Centrality
                    st.markdown("##### Section Contradiction Centrality")
                    st.caption("Sections ranked by how many contradiction edges they have.")
                    centrality = st.session_state.section_centrality or []
                    if centrality:
                        for rank, sec in enumerate(centrality, 1):
                            edges = sec.get("contradiction_edges", 0)
                            label = sec.get("display", sec.get("section", ""))
                            bar_w = min(100, max(5, edges * 10))
                            color = "#ff4444" if edges > 5 else ("#ffaa00" if edges > 2 else "#44cc44")
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:0.8rem;padding:0.3rem 0;">
                                <span style="color:#5555aa;font-size:0.75rem;width:2rem;">#{rank}</span>
                                <span style="color:#c9d1d9;font-size:0.82rem;width:200px;">{label}</span>
                                <div style="background:#14142a;border-radius:4px;flex:1;height:18px;">
                                    <div style="background:{color};width:{bar_w}%;height:100%;border-radius:4px;"></div>
                                </div>
                                <span style="color:#aaaaff;font-size:0.75rem;width:3rem;">{edges}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No section centrality data available.")

                    st.divider()

                    # Entity Clusters
                    st.markdown("##### Entity Contradiction Clusters")
                    st.caption("Topics with the most contradictory statements.")
                    clusters = st.session_state.entity_clusters or []
                    if clusters:
                        for cl_item in clusters[:15]:
                            entity = cl_item.get("entity", "")
                            count = cl_item.get("contradiction_count", 0)
                            samples = cl_item.get("sample_statements", [])
                            with st.expander(f"**{entity}** — {count} contradiction(s)"):
                                for s in samples:
                                    st.markdown(f"- _{s[:150]}{'...' if len(s) > 150 else ''}_")
                    else:
                        st.info("No entity clusters found. Contradictions may not mention identifiable entities.")

                    st.divider()

                    # Transitive Contradictions
                    st.markdown("##### Transitive Contradictions (Multi-hop)")
                    st.caption(
                        "If A contradicts B, and B contradicts C, then A and C "
                        "may also be inconsistent — even if they were never directly compared."
                    )
                    transitives = st.session_state.transitive_contradictions or []
                    if transitives:
                        for t in transitives[:10]:
                            st.markdown(f"""
                            <div class="c-card" style="border-left:3px solid #00dd88;">
                                <div style="font-size:0.78rem;color:#58a6ff;font-weight:600;margin-bottom:0.5rem;">
                                    {t.get('section_a','')} → {t.get('section_bridge','')} → {t.get('section_c','')}
                                </div>
                                <div style="font-size:0.8rem;color:#c9d1d9;margin-bottom:0.3rem;">
                                    <b>A:</b> {t.get('stmt_a','')[:120]}...
                                </div>
                                <div style="font-size:0.8rem;color:#aaa;margin-bottom:0.3rem;">
                                    <b>Bridge:</b> {t.get('bridge','')[:120]}...
                                </div>
                                <div style="font-size:0.8rem;color:#c9d1d9;">
                                    <b>C:</b> {t.get('stmt_c','')[:120]}...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No transitive contradictions detected (this is actually a good sign).")

            # ── EXPORT ───────────────────────────────────────────────────────
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            with e1:
                param_note = ""
                if auto_mode and st.session_state.auto_params:
                    p = st.session_state.auto_params
                    param_note = (f"\nAuto-config: top_k={p['top_k']}, alpha={p['alpha']}, "
                                  f"min_confidence={p['min_confidence']}%\n")
                txt = "\n".join([
                    "CONTRADICTAI REPORT",
                    f"Document: {st.session_state.doc_name}",
                    f"Sections: {len(chunks)}  Contradictions: {len(C)}",
                    param_note,
                    "=" * 60, ""
                ] + [
                    f"#{i+1} | {c['target']} | {c.get('severity')} | {c.get('type')} | {c.get('confidence')}%\n"
                    f"{c['analysis']}\nCompared: {', '.join(c['related_sections'])}\n{'='*60}"
                    for i, c in enumerate(C)
                ])
                st.download_button("📄 Download TXT Report", data=txt,
                    file_name=f"report_{st.session_state.doc_name}.txt",
                    mime="text/plain", use_container_width=True)
            with e2:
                if st.session_state.pdf_bytes:
                    st.download_button("📑 Download PDF Report",
                        data=st.session_state.pdf_bytes,
                        file_name=f"report_{st.session_state.doc_name}.pdf",
                        mime="application/pdf", use_container_width=True)
                else:
                    if st.button("📑 Generate PDF", use_container_width=True):
                        with st.spinner("Generating…"):
                            try:
                                st.session_state.pdf_bytes = generate_pdf_report(
                                    st.session_state.doc_name, st.session_state.doc_name,
                                    len(chunks), C, cl)
                                st.rerun()
                            except Exception as e:
                                st.error(f"PDF error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Q&A CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if st.session_state.chunks and st.session_state.retriever:
        st.markdown(f"""
        <div class="status-bar">
            <div class="status-dot"></div>
            <div class="status-text">
                Chatting about <strong>{st.session_state.doc_name}</strong>
                · {len(st.session_state.chunks)} sections indexed
                · answers grounded in source only
            </div>
        </div>
        """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    tags = "".join(f'<span class="sec-tag">{s}</span>'
                                   for s in msg["sources"])
                    st.markdown(f'<div style="margin-top:0.5rem">{tags}</div>',
                                unsafe_allow_html=True)

        q = st.chat_input("Ask anything about the document…")
        if q:
            with st.chat_message("user"):
                st.markdown(q)
            st.session_state.chat_history.append({"role": "user", "content": q})

            with st.chat_message("assistant"):
                with st.spinner("Searching & reasoning…"):
                    retrieved = st.session_state.retriever.search(q, k=top_k)
                    result    = answer_question(q, retrieved)
                st.markdown(result["answer"])
                tags = "".join(f'<span class="sec-tag">{s}</span>'
                               for s in result["sources"])
                st.markdown(f'<div style="margin-top:0.5rem">{tags}</div>',
                            unsafe_allow_html=True)

            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })

        if st.session_state.chat_history:
            if st.button("Clear chat", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">💬</span>
            <div class="empty-h">No document loaded</div>
            <div class="empty-s">
                Run contradiction detection first — the document will be available here automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)

        qa_upload = st.file_uploader("Or upload directly for Q&A:", type=["pdf"],
                                     key="qa_direct")
        if qa_upload:
            with st.spinner("Processing…"):
                raw    = extract_text_from_pdf(qa_upload.read())
                chunks = label_chunks(semantic_chunk(raw, min_len=min_chunk, max_len=max_chunk))
                emb    = embed_chunks(chunks)
                idx    = build_faiss_index(emb)
                store_chunks(qa_upload.name, chunks, emb)
                st.session_state.chunks     = chunks
                st.session_state.embeddings = emb
                st.session_state.index      = idx
                st.session_state.retriever  = HybridRetriever(chunks, emb, idx)
                st.session_state.doc_name   = qa_upload.name
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPH INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    render_graph_tab()