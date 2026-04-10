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
from pipeline.neo4j_graph import get_graph, NEO4J_AVAILABLE


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
