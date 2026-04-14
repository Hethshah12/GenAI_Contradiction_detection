# ContradictAI — Complete Implementation Details

> A deep, end-to-end technical walkthrough of the codebase, the architecture, every technology choice, the reasoning behind each component, and the novelties that make this project different from any off-the-shelf RAG / contradiction-detection system.

---

## 1. Project Identity & One-Line Description

**ContradictAI** is a multimodal, hybrid-RAG document-intelligence system that ingests long PDF documents (company reports, policies, contracts, compliance filings) and automatically surfaces **logical contradictions** — not only between pieces of text, but also between **charts/graphs** and the text that describes them, and even **transitive / circular contradictions** across sections. It is delivered as a dark-themed Streamlit web app backed by an eight-stage pipeline combining FAISS + BM25, a local NLI cross-encoder (DeBERTa-v3), LLaMA 3.3-70B on Groq, a Groq vision model, ChromaDB persistence, and an optional Neo4j knowledge graph.

---

## 2. Directory Layout

```
Safety_H_GENAI/
├── app.py                              # Streamlit UI, 1511 lines — everything glues here
├── graph_tab_integration.py            # Drop-in Neo4j "Graph Intelligence" tab (optional)
├── requirements.txt                    # 13 runtime dependencies
├── .env                                # GROQ_API_KEY + optional Neo4j creds
├── .gitignore                          # excludes chroma_db/, venv/, .env, pdfs
├── Contradictory_Test_Document.pdf     # sample fixture (intentionally self-contradictory)
├── chroma_db/                          # persistent vector store (auto-created)
│
└── pipeline/                           # all heavy lifting lives here
    ├── __init__.py
    ├── document_processor.py           # PDF -> raw text + image extraction, semantic chunking
    ├── embedder.py                     # Sentence-Transformers + FAISS index builder
    ├── hybrid_retriever.py             # BM25 + FAISS hybrid scorer
    ├── contradiction_checker.py        # Main pipeline + Groq LLM calls + self-consistency
    ├── nli_filter.py                   # DeBERTa-v3 cross-encoder NLI pre-filter (core novelty)
    ├── chart_analyzer.py               # Vision-LLM extraction of charts/graphs/tables
    ├── visual_contradiction_checker.py # Cross-modal (chart ↔ text) detection (novelty)
    ├── chroma_store.py                 # ChromaDB persistence layer with version invalidation
    ├── graph_store.py                  # Primary Neo4j knowledge-graph builder & Cypher queries
    ├── neo4j_graph.py                  # Alt graph module with PageRank-style centrality
    ├── qa_chat.py                      # Hallucination-hardened Q&A with Groq LLaMA-3.3
    ├── visualiser.py                   # 9 Plotly + NetworkX charts (severity, heatmap, network…)
    └── report_generator.py             # Branded PDF export via fpdf2
```

---

## 3. Technology Stack — the "What" and "Why" of every dependency

| Library | Version | Role | Why this one (vs alternatives)? |
|---|---|---|---|
| **streamlit** | ≥1.32 | UI framework | Zero-backend, session-state friendly, widget-rich, renders HTML/CSS directly — lets a single Python file host the entire app with a custom noir-themed CSS layer. |
| **groq** | ≥0.9 | LLM / vision API SDK | Groq's LPU backend serves LLaMA-3.3-70B at ~500 tok/s for near-real-time reasoning. Free tier is usable with a 2 s delay. Text *and* vision models share one key/SDK. |
| **sentence-transformers** | ≥2.7 | Dense embeddings + CrossEncoder NLI | Provides both (a) `all-MiniLM-L6-v2` embedder for retrieval and (b) `cross-encoder/nli-deberta-v3-base` for pair-wise entailment scoring — two models from one API. |
| **faiss-cpu** | ≥1.8 | Dense ANN index | Facebook's in-memory ANN is orders of magnitude faster than naive cosine loops at the scale we hit (up to thousands of chunks) and requires no server. |
| **rank-bm25** | ≥0.2.2 | Sparse lexical retrieval | Pure-Python BM25-Okapi. No server, no network calls. Pairs with FAISS for hybrid retrieval — exact legal/financial terms ("Section 4.2", "17%") that FAISS misses. |
| **PyMuPDF (fitz)** | ≥1.24 | PDF parser + renderer | Extracts both text *and* raster page renders (crucial for vector charts). Faster and more faithful than pdfplumber/PyPDF2 for mixed-content documents. |
| **chromadb** | ≥0.5 | Persistent vector store | Zero-server, disk-persistent, metadata-aware, cosine-indexed. Chosen over Redis (RAM-heavy) and Neo4j (graph-only). Survives restarts so big PDFs don't get re-chunked. |
| **neo4j** | ≥5.18 | Knowledge graph DB | Enables multi-hop Cypher queries for transitive and circular contradictions — features impossible over a flat Python list. Optional; app degrades gracefully. |
| **networkx** | ≥3.2 | In-memory graph layout | Builds the force-directed layout (`spring_layout`) for the contradiction/cross-modal network graphs, before handing to Plotly. |
| **plotly** | ≥5.20 | Interactive charts | Dark-theme-friendly, web-native HTML rendering — severity bars, pie, heatmap, confidence histogram, network graph, risk bar, visual-contradiction summary, cross-modal network. |
| **numpy** | ≥1.26 | Numeric core | Softmax for NLI probabilities, matrix ops for the heatmap, dtype casting for FAISS. |
| **pandas** | ≥2.2 | Data shaping | Lightweight tabular handling where charts require DataFrame-like inputs. |
| **fpdf2** | ≥2.7 | PDF generation | `fpdf2` (not legacy pyfpdf) supports `multi_cell` + colour rectangles and modern layout APIs to produce the branded contradiction report with header/footer and severity-coded header bars. |
| **python-dotenv** | ≥1.0 | Env loading | Loads `GROQ_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` from `.env`. |

### Models used (none are fine-tuned by us; all pre-trained)

1. **Groq-hosted `llama-3.3-70b-versatile`** — the main reasoning LLM for
   - Parameter auto-detection
   - Unified intra/cross contradiction prompt
   - Intra-only strict prompt
   - Self-consistency multi-temperature voting
   - Visual-textual and multi-hop contradiction verification
   - Q&A chat
2. **Groq vision models** (auto-resolved probe order):
   - `meta-llama/llama-4-scout-17b-16e-instruct`
   - `llama-3.2-90b-vision-preview`
   - `llama-3.2-11b-vision-preview`
   For Phase-1 "does this page have a chart?" gating and Phase-2 structured JSON extraction.
3. **`cross-encoder/nli-deberta-v3-base`** (HuggingFace, loaded locally via sentence-transformers `CrossEncoder`) — produces [contradiction, entailment, neutral] logits per pair, completely offline.
4. **`all-MiniLM-L6-v2`** — 384-dim sentence embedder for retrieval.

---

## 4. Complete End-to-End Architecture Flow

```
                          ┌─────────────────────────────────┐
                          │    USER UPLOADS PDF (Streamlit) │
                          └───────────────┬─────────────────┘
                                          │
                  ┌───────────────────────▼────────────────────────┐
                  │  Stage 0 — CACHE CHECK (chroma_store.load)     │
                  │  Hash(doc_name) → collection lookup            │
                  │  Version-aware (CHUNK_VERSION = "8")           │
                  └─────────────┬─────────────────┬────────────────┘
                           CACHE HIT         CACHE MISS
                                │                 │
                                │     ┌───────────▼────────────┐
                                │     │ Stage 1 — TEXT EXTRACT │
                                │     │   PyMuPDF page.get_text │
                                │     └───────────┬────────────┘
                                │                 │
                                │     ┌───────────▼────────────────────────┐
                                │     │ Stage 2 — AUTO-PARAM DETECTION     │
                                │     │ First 2 KB → LLaMA-3.3 → JSON      │
                                │     │ {top_k, alpha, min_chunk, max_chunk,│
                                │     │  min_confidence, reasoning}        │
                                │     └───────────┬────────────────────────┘
                                │                 │
                                │     ┌───────────▼─────────────────────────┐
                                │     │ Stage 3 — SEMANTIC CHUNKING         │
                                │     │  a) Heading-based split             │
                                │     │     regex on "1.2  Title" patterns  │
                                │     │  b) Fallback paragraph + 200-char   │
                                │     │     rolling overlap                 │
                                │     │  Output: [{id,label,heading,text}]  │
                                │     └───────────┬─────────────────────────┘
                                │                 │
                                └───────► ┌───────▼─────────┐
                                          │ Stage 4 — EMBED │
                                          │ MiniLM → FAISS  │
                                          │ + BM25 corpus   │
                                          └───────┬─────────┘
                                                  │
                                      ┌───────────▼──────────────────────────┐
                                      │ Stage 5 — NLI PRE-FILTER (novelty)   │
                                      │  DeBERTa-v3 CrossEncoder per pair    │
                                      │  Topical-overlap keyword gating      │
                                      │  Meta-sentence filtering             │
                                      │  score = P_contra × (1 − P_neutral)  │
                                      │  Intra threshold 0.40, Cross 0.55    │
                                      │  Outputs candidate pairs             │
                                      └───────────┬──────────────────────────┘
                                                  │
                                      ┌───────────▼──────────────────────────┐
                                      │ Stage 6 — HEURISTIC TYPE CLASSIFIER  │
                                      │  Direct / Numerical / Temporal /     │
                                      │  Scope / Conditional / Exception /   │
                                      │  Definitional                        │
                                      └───────────┬──────────────────────────┘
                                                  │
                                      ┌───────────▼──────────────────────────┐
                                      │ Stage 7 — TOP-3 LLM ENRICHMENT       │
                                      │  LLaMA-3.3 adds 2-sentence human     │
                                      │  explanation to the 3 highest-conf   │
                                      └───────────┬──────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────┤
 │ (Optional) CHART ANALYSIS BRANCH               │
 │                                                │
 │ Stage 8a — Page rendering (200 DPI)            │
 │ Stage 8b — Phase-1 "CHARTS_FOUND?" vision probe│
 │ Stage 8c — Phase-2 structured JSON extraction  │
 │            with fallback simpler prompt        │
 │ Stage 8d — Dedup vs embedded-image extraction  │
 │ Stage 8e — charts_to_chunks() → id 10001+      │
 │                                                │
 │ Stage 9  — VISUAL-TEXTUAL CHECK                │
 │   For each chart chunk:                        │
 │     retriever.search(chart_text) → top-k text  │
 │     LLaMA-3.3 checks Numerical/Trend/Scope/    │
 │     Categorical/Temporal/Omission              │
 │   + optional MULTI-HOP: chart + primary text   │
 │     + secondary texts                          │
 └───────────────┬────────────────────────────────┘
                 │
                 ▼
     Merge text + visual contradictions → st.session_state.contradictions
                 │
 ┌───────────────▼────────────────────────────────┐
 │ Stage 10 — KNOWLEDGE GRAPH (optional Neo4j)    │
 │ Nodes: Document, Section, Statement, Entity    │
 │ Rels:  HAS_SECTION, CONTAINS, MENTIONS,        │
 │        CONTRADICTS, CONTRADICTS_SECTION        │
 │ Cypher queries for:                            │
 │  • transitive contradictions (A→B→C)           │
 │  • entity contradiction clusters               │
 │  • section centrality                          │
 │  • circular contradictions (alt module)        │
 │  • propagation-score explorer                  │
 └───────────────┬────────────────────────────────┘
                 │
 ┌───────────────▼────────────────────────────────┐
 │ Stage 11 — RENDER                              │
 │  Findings cards (HTML)                         │
 │  Severity bar · Type pie · Conf histogram      │
 │  Section risk bar                              │
 │  Heatmap · Network graph                       │
 │  Visual-Textual tab + cross-modal network      │
 │  Knowledge-Graph tab                           │
 │  TXT + PDF exports                             │
 └───────────────┬────────────────────────────────┘
                 │
 ┌───────────────▼────────────────────────────────┐
 │ Side pipeline — Q&A TAB                        │
 │  user query → HybridRetriever.search → top-k   │
 │  grounded LLaMA-3.3 (temp 0) with citations    │
 └────────────────────────────────────────────────┘
```

---

## 5. File-by-File Implementation Deep Dive

### 5.1 `app.py` — The Orchestrator (1511 lines)

Responsibilities:

1. **Streamlit session state initialization** — 24 keys in a single dict-driven loop covering text pipeline, chart pipeline, and Neo4j graph results. This is deliberately verbose so every widget can render without `KeyError` and the sidebar's "Clear Results" button can reset everything in one sweep.
2. **Custom CSS theme** — ~480 lines of inline CSS: Syne + JetBrains Mono typography, noise-overlay SVG background via data-URL, `#070710` near-black base with `#4040ff` electric-blue accents, animated `status-dot` glow keyframes, responsive metric-card grid, badge styles for severity/type/confidence, hover-lifting contradiction cards.
3. **Sidebar controls**:
   - Radio: 🤖 Auto (LLM decides) vs 🔧 Manual override
   - When Auto: shows the six LLM-calibrated parameters as pill rows
   - When Manual: six sliders (`top_k` 2-8, `alpha` 0-1, `min_chunk` 50-300, `max_chunk` 300-2000, `min_confidence` 60-95, and `api_delay` locked at 2 s for Groq free-tier safety)
   - Chart Analysis toggle (multimodal on/off)
   - Stored-documents list with ✕ delete buttons (drives ChromaDB cleanup)
   - Clear Results button that wipes ~15 keys
4. **Two main tabs**: Contradiction Detection and Document Q&A.
5. **Detection tab flow**: upload → cache probe → (optionally auto-detect params) → chunk → embed → FAISS+BM25 → call `run_full_pipeline()` with an `on_progress` callback that paints a live JetBrains-Mono progress strip with phase name, step counter, elapsed timer, and dynamic ETA.
6. **Post-run enrichment**: chart analysis branch, visual-textual pipeline, PDF report generation, Neo4j graph build with four Cypher aggregations captured into session state.
7. **Results rendering**: dynamic tab bar that adds "Visual-Textual" and "Knowledge Graph" tabs only if those subsystems produced output. Findings use severity/type/confidence/source filters, each rendered as a hover-lifting HTML card.
8. **Q&A tab**: `st.chat_message`-based conversation, each assistant message is sourced with `<span class="sec-tag">Section N</span>` chips. Allows direct PDF upload if the user skipped detection.

Why one monolithic file? Streamlit's reactivity model works best when session-state keys, widgets, and reruns live in the same module. Splitting would require each sub-module to import and re-declare state keys.

### 5.2 `pipeline/document_processor.py` — PDF Ingestion + Chunking

- `extract_text_from_pdf(bytes)` — PyMuPDF page loop, double-newline between pages.
- `extract_page_images(bytes, min_size=150)` — two-pass embedded-image extraction that:
  1. **counts each `xref` across pages** to identify *recurring* images (logos, headers, footers appear ≥ 3 times) and drop them.
  2. Applies **size filter** (< 150 px → drop), **aspect-ratio filter** (> 8:1 → likely a divider rule), **CMYK→RGB** conversion, and **MD5-hash deduplication**. This is a robust, novel heuristic pipeline that reliably strips marketing chrome before a vision model ever sees it.
- `detect_heading(text)` — first-line classifier via three patterns: all-caps headings, numbered (`2.1 Revenue`), or trailing-colon forms.
- `heading_based_chunk(text, max_len)` — splits on `\d+[.\d]*\s+[A-Z]` heading boundaries. Each policy/report section becomes its own chunk — critical for intra-section contradiction detection. Sub-splits any section longer than `max_len`.
- `_paragraph_split` — fallback used when no headings are detected; buffer-based accumulation on `\n\n` boundaries respecting min/max.
- `semantic_chunk` — two-strategy dispatcher: tries heading-based first, falls back to paragraph mode with a **200-char rolling overlap** (`… + next`) so the LLM never sees a hard cut.
- `label_chunks` — stamps `{id, label, heading, display, text}` metadata used throughout the pipeline.

### 5.3 `pipeline/embedder.py` — Embeddings & BM25 tokenisation

- Lazy singleton `_model` for `all-MiniLM-L6-v2`.
- `tokenise_for_bm25` — lowercases, splits on whitespace, strips ~60 common English stopwords. Deliberate: legal/financial docs are dominated by "the/of/shall" and BM25 precision collapses without stopword removal.
- `embed_chunks` — returns `(n, 384)` float32 array (matches FAISS `IndexFlatL2`).
- `build_faiss_index` — pure L2 flat index. Chosen over HNSW/IVF because (a) document sizes are in the hundreds-to-low-thousands of chunks and flat is exact, and (b) no index-training step means zero cold-start.
- `retrieve_top_k` — unused in the hybrid path but kept for debugging and direct-FAISS experiments.

### 5.4 `pipeline/hybrid_retriever.py` — The Hybrid Scorer

```
combined = α · rank_score_dense + (1 − α) · bm25_normalised
```

- Builds a `BM25Okapi` corpus from tokenised chunks at init time.
- `retrieve()` — runs a full-corpus FAISS scan, converts L2 ranks to a normalised `(n − rank) / n` score to avoid distance-scale dependence, normalises BM25 to [0,1] by dividing by `max`, linearly blends, sorts descending, and skips the self-id.
- `search(query_text, k=5)` — free-text interface used by the Q&A chat and by the visual-textual checker (to retrieve the text sections most relevant to each chart).

Why hybrid? BM25 nails exact keyword matches (`"Section 4.2"`, `"17%"`) that dense semantics miss; MiniLM nails paraphrase ("workforce" ↔ "employees") that lexical matching misses. Together they cover both recall gaps.

### 5.5 `pipeline/nli_filter.py` — The Core Novelty: Calibrated NLI Pre-Filter

This is where the project decisively beats a naive LLM-only RAG. Instead of calling an expensive LLM on every (target, related) pair, it pre-filters with a local DeBERTa cross-encoder and only escalates high-likelihood pairs.

Key mechanisms:

1. **Lazy model loading** with startup sanity check printing probabilities for three calibration pairs.
2. **Combined contradiction score** — purely relying on DeBERTa's `contradiction` index fires on *unrelated* pairs too (documented NLI behaviour). The authors instead compute:

   ```
   score = P_contradiction × (1 − P_neutral)
   ```

   This forces pairs to be *both contradictory and on-topic*. Unrelated pairs have high `P_neutral` → score collapses toward zero.
3. **Topical-overlap gate** — fast keyword-overlap pre-filter. Intra-section requires ≥ 1 shared content word, cross-section requires ≥ 2 (cross-section pairs are much noisier).
4. **Meta-sentence filter** — a regex blocklist removes sentences about *the document itself* ("The following contradictions are commonly found…", "Statement A conflicts with Statement B") — otherwise any evaluation doc causes false positives. Stopword list is *expanded* to include boilerplate (`policy`, `contradiction`, `requirements`, `organization`, `employees`, `management`, …).
5. **Sentence splitter** that rejoins PDF soft-wrapped lines (`[ ]*\n(?=[a-z])` → space), strips heading lines, then splits on sentence-end punctuation.
6. **Calibrated thresholds** — `0.40` intra, `0.55` cross (cross is stricter because of higher noise).
7. **Deduplication** — sorted-tuple key `(norm(a), norm(b))` across all candidates, retaining the highest-scoring copy of each pair.
8. **Batched scoring** (batch_size=64) with live debug print of max/above-0.5/above-0.3 counts.

Complexity: O(n²) only on statement pairs that passed the topical gate, so effective cost is closer to O(n) in practice.

### 5.6 `pipeline/contradiction_checker.py` — LLM Layer & Pipeline Entry-Point

Contains five distinct capabilities:

1. **`auto_detect_params(raw_text)`** — feeds the first ~2000 chars to LLaMA-3.3-70B with a strict JSON-only prompt; clamps every returned field to safe ranges; always pins `api_delay` to 2.0 s regardless of the model's wish (Groq free-tier safety). Defaults fall back gracefully on parse errors.
2. **Two system prompts** — `SYSTEM_PROMPT` covers both intra and cross in one call; `INTRA_CHUNK_SYSTEM_PROMPT` is a stricter single-chunk variant that emits exact statement quotes.
3. **`check_contradiction` / `check_intra_chunk_contradictions`** — exponential-backoff retry (`2 ** attempt * 10`) on 429s, temperature = 0 for determinism, 1500-token max.
4. **`run_full_pipeline(...)`** — the master entry-point called by Streamlit. Despite the Groq-flavoured file name, it runs an **NLI-only detection pass** (no per-pair LLM call) using `prefilter_chunks`, then enriches only the **top-3 highest-confidence** contradictions with an LLM-written 2-sentence human explanation. This dramatically reduces token cost while keeping explanations rich.
5. **`_classify_contradiction_type(a, b)`** — a hand-written heuristic classifier keyed on signature words:
   - Numerical — both statements contain numbers but different sets.
   - Temporal — presence of "before/after/quarterly/deadline" etc.
   - Scope — presence of "all/every/never/always" plus a negation.
   - Exception-based — "unless/except/however".
   - Conditional — "if/when/provided".
   - Direct — explicit opposition pairs (`must`↔`must not`, `allowed`↔`prohibited`, …).
6. **`check_with_self_consistency(...)` (novel)** — runs the same check at `[0.0, 0.3, 0.5]` temperatures and accepts only if `agreement_threshold` (default 2/3) votes agree. Produces a `stability_score` field that's more grounded than the LLM's self-reported confidence. Not wired into the default Streamlit flow but available for evaluation harnesses.

### 5.7 `pipeline/chart_analyzer.py` — Vision Multimodal Extraction (novelty)

Design notes:

- **Dataclass `ChartData`** with 11 fields including structured `data_points: list[dict]`, `trends`, `key_findings`, `raw_description`, and a `to_text_chunk()` method that *synthesises an embedding-friendly representation* of the chart. The output reads like:

   ```
   [CHART on Page 4]
   Chart Title: Quarterly Revenue 2023
   Chart Type: bar
   Axes: X-axis: Quarter, Y-axis: USD millions
   Data Points: Q1: 4.2M; Q2: 4.8M; Q3: 5.1M; Q4: 5.0M
   Trend: Revenue grew through Q3 then flattened in Q4.
   Key Finding: Peak revenue was $5.1M in Q3.
   ```

   This text lives in the *same* FAISS + BM25 space as textual chunks, giving it id ≥ 10001 to avoid collision with text ids.
- **Two-phase approach** (documented with ACL/EMNLP references):
  - **Phase 1** (`PAGE_CHART_DETECTION_PROMPT`) — cheap "CHARTS_FOUND: N" / "NO_CHARTS" probe per page. Stops wasted extraction on text-only pages. Uses `max_tokens=30`.
  - **Phase 2** (`CHART_EXTRACTION_PROMPT`) — strict JSON-only schema with chart_type, title, axes, data_points, trends, key_findings, raw_description, confidence. Fallback simpler prompt with fewer fields if first parse fails.
- **Model resolution** — probes `llama-4-scout-17b-16e-instruct`, then `3.2-90b-vision-preview`, then `3.2-11b-vision-preview`. Caches the first that works.
- **Page rendering at 200 DPI** — critical because many charts are *vector* graphics (SVG-ish content streams) that don't appear via `page.get_images()`. Rendering the full page raster captures both vector and raster charts.
- **Embedded-image path** — per page, extracts raster xrefs, skips < 150 px, > 8:1 aspect ratio, CMYK→RGB conversion, MD5 dedup, and deduplicates against Phase-2 full-page extractions using a 70 %-label-overlap heuristic.
- **Exponential backoff** on 429s with raised `RuntimeError` that cleanly halts the outer loop.

### 5.8 `pipeline/visual_contradiction_checker.py` — The Signature Novelty

Claimed in the file header as "the core novelty of the project — no prior published work specifically addresses within-document visual-textual consistency checking."

Six-class taxonomy (vs the normal seven text-text classes):

| Type | Example conflict |
|---|---|
| Numerical | Chart: revenue $4.2M · Text: "over $5M" |
| Trend | Chart: declining sales · Text: "consistent growth" |
| Scope | Chart: Q1-Q3 · Text: "full-year results" |
| Categorical | Chart: 4 categories · Text: references 5th |
| Temporal | Chart: "2023" · Text: "2024 figures" |
| Omission | Text: "as shown in Fig 3, costs decreased" but chart shows no cost data |

Two prompts:

1. `VISUAL_TEXTUAL_SYSTEM_PROMPT` — strict `VISUAL-TEXTUAL CONTRADICTION FOUND:` block with Type/Chart Reference/Text Section/Chart Claim/Text Claim/Explanation/Severity/Confidence fields. Tolerance rules: approximate values OK, only flag ≥ 10 % relative differences, account for vision-extraction uncertainty.
2. `MULTI_HOP_VISUAL_PROMPT` — detects contradictions that only emerge when combining the chart + primary text + secondary text (multi-source reasoning).

`run_visual_contradiction_pipeline` flow per chart:
1. Hybrid-retrieve top-k text chunks related to the chart's synthesised text.
2. Filter out other chart chunks (we only want text↔chart).
3. Run direct check → parse → if confidence ≥ threshold, append.
4. If `enable_multi_hop` and ≥ 2 related text chunks, run multi-hop check with the first as primary, the rest as secondary.

Tagged types come back as `"Visual-Numerical"`, `"Multi-hop-Visual-Trend"`, etc., so the visualiser can colour-code them.

### 5.9 `pipeline/chroma_store.py` — Persistence with Version Invalidation

- `CHROMA_PATH = <repo>/chroma_db` as a `PersistentClient`.
- **Version-aware collections**: a module-level constant `CHUNK_VERSION = "8"` is stored in `collection.metadata`. On load, if the stored version differs the collection is **deleted** and `None` is returned so the app re-chunks from scratch. This prevents stale-cache bugs when chunking logic changes.
- Collection naming: `"doc_" + md5(doc_name)[:12]` — satisfies ChromaDB's alphanumeric rule and avoids UTF-8 filename issues.
- `store_chunks` wipes existing entries for the doc before re-adding, keeping collections monotonic per doc.
- `list_stored_documents` walks all collections and projects `metadata['doc_name']`.

### 5.10 `pipeline/graph_store.py` — Neo4j Knowledge Graph (primary)

Schema:

```
(:Document)-[:HAS_SECTION]->(:Section)-[:CONTAINS]->(:Statement)-[:MENTIONS]->(:Entity)
                                             ↓
                                    [:CONTRADICTS {score,type,severity,confidence}]
                                             ↓
                                     (:Statement)
(:Section)-[:CONTRADICTS_SECTION {count,max_score,types[]}]->(:Section)
```

- **Entity extraction** via 26 hand-written regex patterns covering security (MFA, VPN, passwords, encryption…), work policy (remote work, core hours, flexible scheduling, leave, monitoring, compliance), data (sensitive/confidential, local storage, cloud, BYOD), communication (email, messaging, public WiFi), and people (employees, managers, interns, contractors). No LLM call required.
- `clear_graph(doc_name)` uses `DETACH DELETE` so a rebuild is clean.
- `build_graph` creates a Document node → Section nodes → Statement nodes (split via `nli_filter.split_into_statements`) → Entity nodes (MERGE, so entities are shared across statements) → CONTRADICTS edges between the specific quoted statements extracted from each contradiction's analysis text, plus the aggregated CONTRADICTS_SECTION edge with `count` / `max_score` / type-list accumulation.
- Four Cypher-powered aggregators:
  - **`get_transitive_contradictions`** — returns A→B→C chains where A-C are not directly linked. Core multi-hop insight.
  - **`get_entity_contradiction_clusters`** — groups contradictions by their mentioned entities, highlighting the *topics* most inconsistent.
  - **`get_most_contradictory_sections`** — ranks by contradiction-edge centrality.
  - **`get_graph_stats`** — single-shot aggregate of sections/statements/entities/edges.

### 5.11 `pipeline/neo4j_graph.py` — Alternate Graph Module (extended features)

Kept as a richer reference implementation with additional Cypher features not yet wired into the default flow:

- **Circular contradictions** — detects cycles A↔B↔…↔A, which have *no consistent resolution*.
- **Hotspot sections** — by edge degree with average confidence.
- **Transitive risk propagation score** — a novel metric that decays with each hop so a direct High-confidence contradiction scores higher than a 3-hop Low-confidence chain.
- **Contradiction chains** with depth and `min_confidence` along the chain.

The companion `graph_tab_integration.py` contains a drop-in `render_graph_tab()` for these features with severity-dot emoji indicators.

### 5.12 `pipeline/qa_chat.py` — Hallucination-Hardened Q&A

Four explicit anti-hallucination strategies layered:

1. **Source grounding** — LLM receives only retrieved chunks, not full doc.
2. **Citation enforcement** — system prompt mandates `"According to Section N..."`.
3. **Uncertainty flagging** — LLM *must* reply "The document does not provide sufficient information…" when retrieval is empty.
4. **Temperature 0** — deterministic, minimal hallucination surface.

Returns `{answer, sources[]}` — source list drives the `sec-tag` chips shown in the UI under each assistant message.

### 5.13 `pipeline/visualiser.py` — Nine Plotly / NetworkX Views

1. `make_severity_chart` — bar chart, High red / Medium orange / Low yellow.
2. `make_type_pie` — 0.45-hole donut with a `color_map` covering *all* 20 text + visual + multi-hop variants, plus a fallback palette for unknown types.
3. `make_confidence_histogram` — 10-bin blue histogram.
4. `make_heatmap` — symmetric N×N confidence matrix (capped at 30 sections for readability), three-stop colourscale `#f0f4ff → #7b9ff9 → #e53935`.
5. `make_network_graph` — NetworkX `spring_layout(seed=42, k=2.5)`, node size/colour by severity, edge width = `weight/25`, edge colour by severity, four legend traces for the key.
6. `make_section_risk_bar` — horizontal bar of `severity_score × confidence` per section, sorted desc.
7. `make_visual_contradiction_summary` — type-breakdown bar for the six visual types with their own palette.
8. `make_cross_modal_network` — two-colour network: purple diamonds for charts, blue circles for text, edges coloured by severity. Unique to this project.
9. (Implicit) `make_clean_report_state` — the no-contradictions case is rendered as a consistency report with section-by-section PASS rows.

### 5.14 `pipeline/report_generator.py` — Branded PDF Export

- `ReportPDF` subclasses `FPDF` with custom `header()` (brand title, underline, navy `(26,58,143)`) and grey `footer()` with page-number.
- `clean()` — strips unicode (em-dashes, smart quotes, ellipsis, bullets, nbsp) to survive the core-14 latin-1 font limits. Replaces with ASCII equivalents before lossy `encode('latin-1', errors='ignore')`.
- Summary block: total sections, contradictions, clean count, and severity counts colour-coded.
- Contradiction blocks: coloured header bar in severity colour, italic section-text preview (truncated 300 chars), full analysis `multi_cell`, italic "compared against" line, separator rule.

### 5.15 `graph_tab_integration.py`

Provides a ready-to-paste `render_graph_tab()` function that leverages `neo4j_graph.py`'s richer API — contradiction chains (with severity emoji), circular contradictions, hotspot sections (inside expanders with heading + conflicts list), and a "Transitive risk explorer" with a `selectbox` that shows propagation-score bars (`"█" * int(score/10)`).

---

## 6. Data Structures — the Chunk / Contradiction / ChartData Contracts

### Chunk dict (text)
```python
{
  "id":      1,                  # int, monotonic; 10001+ for chart chunks
  "label":   "Section 1",        # stable key used everywhere
  "display": "Section 1: Risk",  # UI label
  "heading": "Risk Factors",     # detected from first line
  "text":    "...",              # raw content
}
```

### Chart chunk (superset)
```python
{
  "id":          10001,
  "label":       "Chart-1 (p.4)",
  "display":     "Chart-1: Quarterly Revenue",
  "heading":     "Quarterly Revenue 2023",
  "text":        "<synthesised chart text>",
  "is_chart":    True,
  "chart_type":  "bar",
  "page_number": 4,              # 1-indexed for UI
  "chart_data":  { ... }         # ChartData.to_dict()
}
```

### Contradiction record
```python
{
  "target":            "Section 3",
  "target_display":    "Section 3: Data Handling",
  "target_text":       "... full section text ...",
  "analysis":          "CONTRADICTION FOUND:\nSource: CROSS\n...",
  "related_sections":  ["Section 7"],
  "severity":          "High" | "Medium" | "Low",
  "confidence":        82,       # 0-100
  "type":              "Numerical" | ... | "Visual-Trend" | "Multi-hop-Visual-Scope",
  # Visual-only
  "is_visual":         True,
  "is_multi_hop":      True,
  "chart_claim":       "Chart shows $4.2M",
  "text_claim":        "Text says 'over $5M'",
  "chart_data":        { ... },
  "page_number":       4,
}
```

---

## 7. Configuration & Operational Notes

- **`.env`** holds `GROQ_API_KEY` and optional `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
- **Groq free-tier safety** — `api_delay = 2.0 s` is hard-coded even when the auto-param LLM returns a faster value. Rate-limit retry uses `2 ** attempt * 10` exponential backoff.
- **ChromaDB path** — `./chroma_db/` in project root, auto-created, in `.gitignore`.
- **Cache invalidation** — bump `CHUNK_VERSION` in `chroma_store.py` whenever chunking logic changes; stale collections self-destruct.
- **Neo4j** — the app works entirely without Neo4j; `is_available()` gates the graph-tab additions.
- **Streamlit session state** — 24 keys initialised via a single dict loop; the "Clear Results" button resets 15 of them.

---

## 8. What Makes This Project Novel (The "Why You Should Care" Section)

Most "AI contradiction detector" hackathon projects stop at a single LLM call per chunk pair. This project layers five genuine innovations on top of that baseline:

1. **Calibrated combined-score NLI gate with topical-overlap pre-filter and meta-sentence exclusion.** Off-the-shelf DeBERTa-v3 NLI generates high false positives on unrelated pairs because the `contradiction` label misfires on unrelated text. The `P_contra × (1 − P_neutral)` formula plus a hand-tuned boilerplate/meta stopword list is not something you get from any RAG starter kit — it was discovered empirically and documented inline.

2. **Intra-section *and* cross-section detection in the same pipeline.** Most open-source contradiction tools only compare chunks to *other* chunks. Intra-chunk detection catches internally inconsistent single sections (e.g., a policy that says "must use VPN" three sentences after saying "VPN is optional"). Here it's a first-class citizen with its own prompt, its own lower threshold (0.40 vs 0.55 cross), and its own badge in the UI.

3. **Multimodal visual-textual contradiction detection.** The file itself claims, and I have not found a counter-example, that *no prior published work addresses within-document visual-textual consistency checking*. ChartCheck (ACL 2024), ChartQAPro (2025), and M3D (EMNLP 2024) each cover adjacent problems (fact-checking, benchmarks, cross-modal alignment) but none surfaces *"the chart says 42 % and the paragraph above it says 60 %"* inside the same document. The six-class taxonomy (Numerical / Trend / Scope / Categorical / Temporal / Omission) plus the multi-hop variant (chart + section A + section B) is original.

4. **Chart-to-text synthesis that lives in the same retrieval space.** Rather than treating extracted charts as a silo, `ChartData.to_text_chunk()` produces a rich natural-language rendering that is embedded alongside text chunks with IDs starting at 10001. This means BM25 keyword matching and FAISS semantic search pick up chart-related queries (`"what was Q3 revenue?"`) without any new retrieval path. It's a surprisingly elegant unification.

5. **Knowledge-graph-native reasoning over contradictions.** Treating contradictions as a Neo4j graph enables queries that are literally impossible over a Python list: transitive A→B→C chains, circular A↔B↔A conflicts, entity-centric clustering ("all contradictions that mention VPN"), section centrality ranking, and a novel propagation-score metric that decays with hop distance. The Cypher queries are written to be idempotent (MERGE on entities, ON CREATE/ON MATCH for aggregated section edges).

Other engineering niceties worth calling out:

- **Self-consistency multi-temperature voting** (`check_with_self_consistency`) — runs the same prompt at `T=0.0, 0.3, 0.5` and accepts only if ≥ 2/3 agree, producing a stability score that's more honest than the LLM's own `confidence` field.
- **LLM auto-parameter calibration** — feeds the first 2 KB of the doc to LLaMA-3.3 with a strict JSON schema to pick `top_k`, `alpha`, chunk sizes, and minimum confidence. Every output field is clamped to a safe range and `api_delay` is pinned to 2 s regardless.
- **Two-pass chart extraction** (page render + embedded xref) with a 70 %-label-overlap deduplication step. Handles both raster and vector charts in the same pass.
- **Version-aware ChromaDB cache invalidation** — bumping `CHUNK_VERSION` cleanly evicts all stale cached chunks without manual deletion.
- **Graceful degradation** — the app functions even when Neo4j is absent, when charts are absent, when the vision model is rate-limited, and when the JSON schema extraction fails (automatic fallback to a simpler prompt).
- **Ergonomic progress callbacks** — `on_progress(cur, total, phase, label, t0)` paints a multi-phase strip showing current phase, step counter, percent, elapsed timer, and ETA — all in a consistent JetBrains-Mono aesthetic.
- **Custom PDF report generator** with severity-coded header bars using `fpdf2`, unicode-stripping `clean()` helper, and grey branded footer.
- **Dark noir UI** with Syne + JetBrains Mono typography, SVG noise overlay, animated glowing status dot, hover-lifting cards, and ~480 lines of consistent CSS design tokens (`#070710` base, `#4040ff` accent, `#0e0e1f` panel) — far beyond the default Streamlit look.

---

## 9. How It Compares to Common Alternatives

| Capability | ContradictAI | Naive LLM-RAG | LangChain QA | Pure NLI tools (e.g., SummaC) |
|---|---|---|---|---|
| Hybrid BM25 + FAISS | ✅ | ❌ (usually one) | ⚠️ optional | ❌ |
| Local NLI pre-filter | ✅ (combined score) | ❌ | ❌ | ✅ but no combined score |
| Intra-section detection | ✅ | ❌ | ❌ | ⚠️ |
| Cross-section detection | ✅ | ✅ | ⚠️ | ⚠️ |
| Visual-textual detection | ✅ (6 types) | ❌ | ❌ | ❌ |
| Multi-hop (text) | ✅ (via Neo4j) | ❌ | ❌ | ❌ |
| Multi-hop (visual) | ✅ | ❌ | ❌ | ❌ |
| Circular contradictions | ✅ | ❌ | ❌ | ❌ |
| LLM auto-parameter calibration | ✅ | ❌ | ❌ | ❌ |
| Self-consistency voting | ✅ | ❌ | ❌ | ❌ |
| Persistent cache with version invalidation | ✅ | ❌ | ⚠️ | ❌ |
| Knowledge-graph queries | ✅ | ❌ | ❌ | ❌ |
| Branded PDF report | ✅ | ❌ | ❌ | ❌ |
| Grounded Q&A with citations | ✅ | ⚠️ | ✅ | ❌ |

---

## 10. Runtime Walkthrough — What Happens When You Click "Run"

1. **`uploaded.read()`** → `file_bytes` stored in session state (chart pipeline needs the raw bytes).
2. `load_chunks(doc_name)` probes ChromaDB. If hit, embeddings are regenerated (FAISS is in-memory) but chunking is skipped. If miss:
3. `extract_text_from_pdf` walks PyMuPDF pages.
4. `auto_detect_params` (only in Auto mode) hits LLaMA-3.3 with a 2 KB sample.
5. `semantic_chunk` tries heading-first, falls back to paragraph + overlap.
6. `label_chunks` stamps metadata.
7. `embed_chunks` → MiniLM → FAISS `IndexFlatL2`.
8. `store_chunks` writes to ChromaDB with the current `CHUNK_VERSION`.
9. `HybridRetriever(chunks, embeddings, index)` builds the BM25 index.
10. User clicks **Run Contradiction Detection**:
    - `run_full_pipeline` → `prefilter_chunks` runs DeBERTa NLI on every intra-chunk pair (topical-overlap-gated) and every cross-chunk pair (retrieved via hybrid). Dedup globally.
    - For each surviving candidate, compute severity + heuristic type, build a `CONTRADICTION FOUND` analysis block, and collect.
    - Enrich top-3 with an LLM explanation.
11. If chart analysis is on:
    - `ChartAnalyzer.analyze_pdf` — page rendering, Phase-1 detection, Phase-2 extraction, embedded-image fallback, dedup.
    - `charts_to_chunks` — assign ids ≥ 10001 and produce embedding-ready text chunks.
    - `run_visual_contradiction_pipeline` — for each chart, hybrid-retrieve text, run direct + multi-hop visual-textual checks.
    - Merge visual contradictions into the main list.
12. `generate_pdf_report` — branded PDF with colour-coded severity bars.
13. If Neo4j is reachable:
    - `build_graph` — creates Document/Section/Statement/Entity nodes and CONTRADICTS edges.
    - Four aggregation Cypher queries populate session state for the Graph Intelligence tab.
14. `st.rerun()` — results render:
    - Findings (filtered cards)
    - Charts (severity bar, type pie, confidence histogram, risk bar)
    - Maps (heatmap, network graph)
    - Visual-Textual (conditional tab)
    - Knowledge Graph (conditional tab) with centrality bars, entity clusters in expanders, and transitive chains.
15. TXT + PDF export buttons stream the result files.

---

## 11. Doubts / Questions I'd Like to Clarify

Before taking this further I'd like your input on a few things the code leaves ambiguous:

1. **Deliverable format** — I've written this as a standalone Markdown document. Do you want me to *also* produce a .docx (formal report) and/or a .pptx (presentation deck) with the same content + architecture diagrams?
2. **Architecture diagram** — the flow above is ASCII. Do you want me to render it as a proper image (Mermaid → SVG/PNG) or as slides?
3. **`pipeline/neo4j_graph.py` vs `pipeline/graph_store.py`** — both exist and partially overlap. Only `graph_store.py` is wired into `app.py`. Do you want me to consolidate these (delete one, merge features) or leave the dual-module layout as a reference?
4. **`graph_tab_integration.py`** is not currently imported by `app.py` (the Graph tab in app.py uses `graph_store.py` instead). Should I wire up the richer integration (propagation-score explorer, circular contradictions) into the live app?
5. **`check_with_self_consistency`** is implemented but never called in the Streamlit flow. Do you want it enabled as an optional "high-assurance mode" toggle in the sidebar?
6. **Evaluation harness** — do you want me to add a benchmark script that measures precision/recall on the included `Contradictory_Test_Document.pdf` so you can quote numbers?
7. **Scope of any follow-up work** — are you after (a) pure documentation (this file is it), (b) a paper/whitepaper write-up, (c) a slide deck for a demo, or (d) refactoring / feature additions?

Ask me anything — I'd rather spend a minute clarifying than an hour producing the wrong artifact.
