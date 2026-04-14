# ─────────────────────────────────────────────────────────────────────────────
# HOW TO INTEGRATE THE GRAPH TAB INTO app.py
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Add this import near the top of app.py:
#
#    from pipeline.neo4j_graph import get_graph
#
# 2. In the section where you define your tab bar (look for st.tabs),
#    add "Graph Intelligence" as a new tab.
#
# 3. Paste the render_graph_tab() function below into app.py,
#    then call it inside the new tab block.
#
# 4. After run_full_pipeline() completes, add one call to push results
#    into Neo4j:
#
#    graph = get_graph()
#    if graph:
#        graph.clear_document(st.session_state.doc_name)
#        graph.store_all(
#            st.session_state.doc_name,
#            st.session_state.chunks,
#            st.session_state.contradictions
#        )
#        graph.close()
#
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

# Backwards-compat: pipeline.neo4j_graph was merged into pipeline.graph_store.
# Import from the consolidated module but keep the old path working.
try:
    from pipeline.graph_store import get_graph, NEO4J_AVAILABLE
except ImportError:
    from pipeline.neo4j_graph import get_graph, NEO4J_AVAILABLE  # type: ignore


def render_graph_tab():
    """
    Graph Intelligence tab — shows Neo4j-powered insights that a flat
    contradiction list cannot provide.
    """
    st.markdown("### Graph Intelligence")
    st.markdown(
        "These insights come from treating contradictions as a **graph** — "
        "sections are nodes, contradictions are directed edges. "
        "None of this is possible with a flat list."
    )

    if not NEO4J_AVAILABLE:
        st.warning(
            "Neo4j package not installed. Run `pip install neo4j` then "
            "add your connection details to `.env`."
        )
        st.code(
            "NEO4J_URI=bolt://localhost:7687\n"
            "NEO4J_USERNAME=neo4j\n"
            "NEO4J_PASSWORD=yourpassword",
            language="bash"
        )
        return

    if not st.session_state.get("analysis_done"):
        st.info("Run contradiction analysis first to populate the graph.")
        return

    graph = get_graph()
    if graph is None:
        st.error(
            "Cannot connect to Neo4j. Check your URI and credentials in `.env`."
        )
        return

    doc_name = st.session_state.doc_name

    # ── Summary metrics ───────────────────────────────────────────────────────
    summary = graph.get_graph_summary(doc_name)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sections in graph", summary["sections"])
    c2.metric("Contradiction edges", summary["edges"])
    c3.metric("Avg confidence", f"{summary['avg_confidence']}%")

    st.divider()

    # ── Interactive contradiction network ─────────────────────────────────────
    st.markdown("#### Interactive contradiction network")
    st.caption(
        "Each node is a section; each edge is a detected contradiction. "
        "Node size = number of conflicts the section is involved in · "
        "edge color = severity · hover a node or edge for details."
    )
    _render_interactive_graph(graph, doc_name)

    st.divider()

    # ── Contradiction chains ──────────────────────────────────────────────────
    st.markdown("#### Contradiction chains")
    st.caption(
        "Paths where A contradicts B, B contradicts C, etc. "
        "Each chain represents a connected web of inconsistency."
    )
    chains = graph.get_contradiction_chains(doc_name)
    if chains:
        for chain_data in chains[:8]:
            depth = chain_data["depth"]
            conf = chain_data["min_confidence"]
            labels = " → ".join(chain_data["chain"])
            severity_color = "🔴" if conf >= 80 else "🟠" if conf >= 65 else "🟡"
            st.markdown(
                f"{severity_color} **Depth {depth}** · min confidence {conf}%  \n"
                f"`{labels}`"
            )
    else:
        st.success("No multi-hop contradiction chains found.")

    st.divider()

    # ── Circular contradictions ───────────────────────────────────────────────
    st.markdown("#### Circular contradictions")
    st.caption(
        "These are the most dangerous: A contradicts B, "
        "B contradicts A (or through more sections). "
        "There is no consistent resolution."
    )
    loops = graph.get_contradiction_loops(doc_name)
    if loops:
        st.error(f"Found {len(loops)} circular contradiction(s)!")
        for loop_data in loops:
            labels = " → ".join(loop_data["loop"])
            st.markdown(f"- `{labels}`")
    else:
        st.success("No circular contradictions found.")

    st.divider()

    # ── Hotspot sections ──────────────────────────────────────────────────────
    st.markdown("#### Hotspot sections")
    st.caption(
        "Sections with the most contradiction edges — "
        "the highest-risk areas of the document."
    )
    hotspots = graph.get_hotspot_sections(doc_name, top_n=8)
    if hotspots:
        for h in hotspots:
            with st.expander(
                f"**{h['section']}** — {h['degree']} conflicts "
                f"(avg {round(h['avg_confidence'])}% confidence)"
            ):
                st.markdown(f"**Heading:** {h['heading']}")
                st.markdown(
                    "**Conflicts with:** " +
                    ", ".join(f"`{s}`" for s in h["conflicts_with"])
                )

    st.divider()

    # ── Transitive risk explorer ──────────────────────────────────────────────
    st.markdown("#### Transitive risk explorer")
    st.caption(
        "Pick a section to see its *propagation score* — a novel metric "
        "that decays with each hop so a direct High-confidence contradiction "
        "scores higher than a 3-hop Low-confidence chain. "
        "No other document tool computes this."
    )

    section_labels = [c["label"] for c in st.session_state.chunks]
    selected = st.selectbox("Select a section to explore", section_labels)

    if selected:
        risks = graph.get_transitive_risk(selected, doc_name)
        if risks:
            st.markdown(
                f"**{selected}** can reach {len(risks)} other sections "
                f"through contradiction chains:"
            )
            for r in risks[:10]:
                score = r["propagation_score"]
                hops = r["hops"]
                path = " → ".join(r["path_labels"])
                bar = "█" * max(1, int(score / 10))
                st.markdown(
                    f"- **{r['section']}** · {hops} hop(s) · "
                    f"score `{score}` `{bar}`  \n"
                    f"  path: `{path}`"
                )
        else:
            st.info(f"{selected} has no transitive contradictions.")

    graph.close()


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE NETWORK VIZ (Plotly + NetworkX spring layout)
# ─────────────────────────────────────────────────────────────────────────────
# Renders the section-level contradiction graph as an interactive Plotly
# figure. Lazy imports keep the main pipeline unaffected if plotly or
# networkx are missing — the tab degrades to a textual fallback.

_SEVERITY_COLOR = {
    "High":   "#d73030",
    "Medium": "#f08819",
    "Low":    "#b59710",
}


def _render_interactive_graph(graph, doc_name: str) -> None:
    """Draw the Neo4j-derived contradiction graph as an interactive plot."""
    # Lazy imports so the tab still loads if plotly/networkx aren't installed.
    try:
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError:
        st.info(
            "Interactive graph requires `plotly` and `networkx`. "
            "Install with: `pip install plotly networkx`."
        )
        return

    try:
        data = graph.get_graph_nodes_and_edges(doc_name)
    except Exception as e:
        st.warning(f"Could not load graph data: {e}")
        return

    nodes = data.get("nodes") or []
    edges = data.get("edges") or []

    if not nodes:
        st.info("No sections found in the graph for this document.")
        return
    if not edges:
        st.success(
            "No contradiction edges — this document appears internally "
            "consistent at the section level."
        )
        return

    # Build NetworkX graph for layout. We treat edges as undirected for
    # layout purposes (contradictions are symmetric logically).
    G = nx.Graph()
    node_by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        G.add_edge(e["source"], e["target"], **e)

    # Only render connected nodes + first-degree neighbours. Completely
    # isolated sections would just float in the corners and distract.
    connected_ids = set()
    for e in edges:
        connected_ids.add(e["source"])
        connected_ids.add(e["target"])
    if not connected_ids:
        st.info("Graph has nodes but no edges to display.")
        return

    G_sub = G.subgraph(connected_ids).copy()

    # Seed the layout deterministically so the picture is stable on reruns.
    try:
        pos = nx.spring_layout(G_sub, k=1.0 / max(1, len(G_sub) ** 0.5),
                               iterations=60, seed=42)
    except Exception:
        pos = nx.kamada_kawai_layout(G_sub)

    # Edge traces — one per severity level so the legend reads cleanly.
    severity_traces: dict[str, dict] = {
        "High":   {"x": [], "y": [], "text": []},
        "Medium": {"x": [], "y": [], "text": []},
        "Low":    {"x": [], "y": [], "text": []},
    }
    for e in edges:
        if e["source"] not in pos or e["target"] not in pos:
            continue
        sev = e.get("severity") or "Low"
        bucket = severity_traces.setdefault(
            sev, {"x": [], "y": [], "text": []}
        )
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        bucket["x"] += [x0, x1, None]
        bucket["y"] += [y0, y1, None]
        bucket["text"].append(
            f"{e['source']} ↔ {e['target']}<br>"
            f"Severity: {sev}<br>Confidence: {e.get('confidence', 0)}%<br>"
            f"Type: {e.get('type', 'Direct')}"
        )

    fig = go.Figure()

    # Edges
    for sev, bucket in severity_traces.items():
        if not bucket["x"]:
            continue
        fig.add_trace(go.Scatter(
            x=bucket["x"], y=bucket["y"],
            mode="lines",
            line=dict(
                color=_SEVERITY_COLOR.get(sev, "#888"),
                width=3 if sev == "High" else 2 if sev == "Medium" else 1.2,
            ),
            hoverinfo="skip",
            name=f"{sev} severity ({len(bucket['text'])})",
            showlegend=True,
            opacity=0.75,
        ))

    # Nodes — size scaled by degree, colored by top-edge severity
    node_x, node_y, node_text, node_labels = [], [], [], []
    node_sizes, node_colors = [], []
    for nid in G_sub.nodes():
        if nid not in pos:
            continue
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(y)
        meta = node_by_id.get(nid, {})
        degree = G_sub.degree(nid)
        # Highest severity adjacent edge colours the node
        best_sev = "Low"
        for _, _, data_ in G_sub.edges(nid, data=True):
            sev = data_.get("severity", "Low")
            if sev == "High" or (sev == "Medium" and best_sev == "Low"):
                best_sev = sev
                if sev == "High":
                    break
        node_colors.append(_SEVERITY_COLOR.get(best_sev, "#888"))
        node_sizes.append(14 + degree * 6)
        node_labels.append(nid)
        preview = (meta.get("preview") or "")[:140].replace("\n", " ")
        heading = meta.get("heading") or ""
        node_text.append(
            f"<b>{nid}</b><br>"
            f"{heading}<br><br>"
            f"Conflicts: {degree}<br>"
            f"{preview}…"
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color="#1a1a2a", width=1.5),
            opacity=0.95,
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=10, color="#e6edf3"),
        hoverinfo="text",
        hovertext=node_text,
        name="Sections",
        showlegend=False,
    ))

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(17, 23, 37, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#e6edf3"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Quick stats under the graph
    sev_counts = data.get("severity_counts") or {}
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Nodes", len(G_sub.nodes()))
    s2.metric("Edges", len(edges))
    s3.metric("High-sev edges", sev_counts.get("High", 0))
    s4.metric(
        "Most connected",
        max(G_sub.degree, key=lambda x: x[1])[0] if G_sub.edges else "—"
    )


# ─────────────────────────────────────────────────────────────────────────────
# WHERE IN app.py TO CALL render_graph_tab()
# ─────────────────────────────────────────────────────────────────────────────
#
# Find where your existing tabs are defined. It will look something like:
#
#   tab1, tab2, tab3 = st.tabs(["Overview", "Contradictions", "Q&A"])
#   with tab1:
#       ...
#   with tab2:
#       ...
#
# Change it to:
#
#   tab1, tab2, tab3, tab4 = st.tabs([
#       "Overview", "Contradictions", "Q&A", "Graph Intelligence"
#   ])
#   with tab4:
#       render_graph_tab()
#
