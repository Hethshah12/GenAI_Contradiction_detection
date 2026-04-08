import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st

from pipeline.document_processor import extract_text_from_pdf, semantic_chunk, label_chunks
from pipeline.embedder import embed_chunks, build_faiss_index
from pipeline.hybrid_retriever import HybridRetriever
from pipeline.contradiction_checker import run_full_pipeline, auto_detect_params
from pipeline.chroma_store import store_chunks, load_chunks, list_stored_documents, delete_document
from pipeline.qa_chat import answer_question
from pipeline.visualiser import (
    make_severity_chart, make_type_pie,
    make_confidence_histogram, make_heatmap,
    make_network_graph, make_section_risk_bar
)
from pipeline.report_generator import generate_pdf_report

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
    "chunks":          None,
    "embeddings":      None,
    "index":           None,
    "retriever":       None,
    "contradictions":  None,
    "clean_count":     None,
    "heatmap_data":    None,
    "doc_name":        None,
    "analysis_done":   False,
    "chat_history":    [],
    "pdf_bytes":       None,
    "auto_params":     None,
    "raw_text":        None,
    "params_detected": False,
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
                      "pdf_bytes", "auto_params", "raw_text", "params_detected"]:
                st.session_state[k] = None if k != "analysis_done" else False
                if k == "params_detected":
                    st.session_state[k] = False
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
tab_detect, tab_chat = st.tabs(["⚡  Contradiction Detection", "💬  Document Q&A"])

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
        est    = round((len(chunks) * api_delay) / 60, 1)

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

                prog   = st.progress(0)
                status = st.empty()

                def on_progress(cur, total, label):
                    prog.progress(int((cur + 1) / total * 100))
                    left = round((total - cur - 1) * api_delay)
                    status.caption(f"→ {label}  ·  {cur+1}/{total}  ·  ~{left}s left")

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
                status.empty()

                st.session_state.contradictions = c
                st.session_state.clean_count    = cl
                st.session_state.heatmap_data   = hm
                st.session_state.analysis_done  = True

                try:
                    st.session_state.pdf_bytes = generate_pdf_report(
                        doc_name, doc_name, len(chunks), c, cl)
                except Exception:
                    st.session_state.pdf_bytes = None

                st.rerun()

    # ── RESULTS ──────────────────────────────────────────────────────────────
    if st.session_state.analysis_done:
        C      = st.session_state.contradictions
        cl     = st.session_state.clean_count
        chunks = st.session_state.chunks or []
        hm     = st.session_state.heatmap_data or {}

        high   = sum(1 for c in C if c.get("severity") == "High")
        medium = sum(1 for c in C if c.get("severity") == "Medium")
        low    = sum(1 for c in C if c.get("severity") == "Low")

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
                <div class="m-sub">detected</div>
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

        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                    color:#5050a0;letter-spacing:0.1em;margin-bottom:1.5rem;'>
            ◈ {st.session_state.doc_name or 'document'}
            &nbsp;·&nbsp; {len(chunks)} sections
            &nbsp;·&nbsp; min confidence {min_conf}%
        </div>
        """, unsafe_allow_html=True)

        if not C:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-icon">✅</span>
                <div class="empty-h">No contradictions found</div>
                <div class="empty-s">The document appears internally consistent above the confidence threshold.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            r1, r2, r3 = st.tabs(["📋  Findings", "📊  Charts", "🗺️  Maps"])

            # ── FINDINGS ─────────────────────────────────────────────────────
            with r1:
                f1, f2 = st.columns(2)
                sev_f = f1.selectbox("Severity", ["All", "High", "Medium", "Low"], key="sf")
                typ_f = f2.selectbox("Type", ["All", "Direct", "Conditional", "Exception-based"], key="tf")

                shown = 0
                for i, c in enumerate(C):
                    sev   = c.get("severity", "Low")
                    ctype = c.get("type", "Direct")
                    conf  = c.get("confidence", 0)

                    if sev_f != "All" and sev != sev_f:     continue
                    if typ_f != "All" and ctype != typ_f:   continue

                    badge_sev = {"High": "b-high", "Medium": "b-medium", "Low": "b-low"}.get(sev, "b-low")

                    st.markdown(f"""
                    <div class="c-card">
                        <div class="c-card-header">
                            <div class="c-num">#{i+1:02d}</div>
                            <div class="c-title">{c.get('target_display', c['target'])}</div>
                            <span class="badge {badge_sev}">{sev}</span>
                            <span class="badge b-type">{ctype}</span>
                            <span class="badge b-conf">{conf}%</span>
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