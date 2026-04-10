import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd


def severity_to_score(severity: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(severity, 1)


def make_severity_chart(contradictions: list) -> go.Figure:
    """Bar chart of contradiction counts by severity."""
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for c in contradictions:
        counts[c.get("severity", "Low")] += 1

    colors = {"High": "#e53935", "Medium": "#fb8c00", "Low": "#fdd835"}

    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=[colors[k] for k in counts.keys()],
        text=list(counts.values()),
        textposition="outside"
    ))
    fig.update_layout(
        title="Contradictions by Severity",
        xaxis_title="Severity",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
        height=350,
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def make_type_pie(contradictions: list) -> go.Figure:
    """Pie chart of contradiction types (text + visual)."""
    counts = {}
    for c in contradictions:
        t = c.get("type", "Direct")
        counts[t] = counts.get(t, 0) + 1

    labels = [k for k, v in counts.items() if v > 0]
    values = [v for k, v in counts.items() if v > 0]

    # Dynamic color palette covering all possible types
    color_map = {
        "Direct": "#1a3a8f", "Conditional": "#2d5be3", "Exception-based": "#7b9ff9",
        "Numerical": "#e53935", "Temporal": "#42a5f5", "Definitional": "#66bb6a",
        "Scope": "#fdd835",
        "Visual-Numerical": "#ff6659", "Visual-Trend": "#fb8c00",
        "Visual-Scope": "#ffee58", "Visual-Categorical": "#81c784",
        "Visual-Temporal": "#64b5f6", "Visual-Omission": "#ce93d8",
        "Multi-hop-Visual-Numerical": "#d32f2f", "Multi-hop-Visual-Trend": "#e65100",
        "Multi-hop-Visual-Scope": "#f9a825", "Multi-hop-Visual-Categorical": "#388e3c",
        "Multi-hop-Visual-Temporal": "#1565c0", "Multi-hop-Visual-Unknown": "#7b1fa2",
    }
    palette_fallback = ["#8888ff", "#aaaaff", "#ccccff", "#6666cc", "#4444aa"]
    colors = []
    for i, label in enumerate(labels):
        colors.append(color_map.get(label, palette_fallback[i % len(palette_fallback)]))

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.45,
        textinfo="label+percent"
    ))
    fig.update_layout(
        title="Contradiction Types",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig


def make_confidence_histogram(contradictions: list) -> go.Figure:
    """Histogram of confidence scores."""
    scores = [c.get("confidence", 75) for c in contradictions]

    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=10,
        marker_color="#2d5be3",
        opacity=0.85
    ))
    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence (%)",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def make_heatmap(chunks: list, contradictions: list) -> go.Figure:
    """
    Contradiction heatmap: sections on both axes,
    colour = confidence score where contradictions exist.
    """
    n = min(len(chunks), 30)  # Cap at 30 sections for readability
    labels = [c["label"] for c in chunks[:n]]

    matrix = np.zeros((n, n))
    label_index = {label: i for i, label in enumerate(labels)}

    for c in contradictions:
        t_label = c["target"]
        conf = c.get("confidence", 75)
        for r_label in c["related_sections"]:
            ti = label_index.get(t_label)
            ri = label_index.get(r_label)
            if ti is not None and ri is not None:
                matrix[ti][ri] = conf
                matrix[ri][ti] = conf  # symmetric

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#f0f4ff"],
            [0.5, "#7b9ff9"],
            [1.0, "#e53935"]
        ],
        zmin=0,
        zmax=100,
        colorbar=dict(title="Confidence %"),
        hoverongaps=False,
        hovertemplate="Section A: %{y}<br>Section B: %{x}<br>Confidence: %{z}%<extra></extra>"
    ))
    fig.update_layout(
        title="Contradiction Heatmap (Section Pairs)",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(t=60, b=120, l=100, r=40)
    )
    return fig


def make_network_graph(chunks: list, contradictions: list) -> go.Figure:
    """
    Network graph where nodes = sections, edges = contradictions.
    Edge thickness = confidence, node colour = severity level.
    """
    G = nx.Graph()

    # Add all chunk nodes
    for chunk in chunks:
        G.add_node(chunk["label"], heading=chunk.get("heading", chunk["label"]))

    # Add edges for contradictions
    severity_map = {}
    for c in contradictions:
        t = c["target"]
        conf = c.get("confidence", 75)
        sev = c.get("severity", "Low")
        severity_map[t] = sev
        for r in c["related_sections"]:
            if G.has_node(r):
                G.add_edge(t, r, weight=conf, severity=sev)

    if len(G.nodes) == 0:
        return go.Figure()

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Edges
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 50)
        sev = data.get("severity", "Low")
        color = {"High": "#e53935", "Medium": "#fb8c00", "Low": "#fdd835"}.get(sev, "#aaa")
        width = max(1, weight / 25)

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none",
            showlegend=False
        ))

    # Nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    contradiction_nodes = set(c["target"] for c in contradictions)

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        heading = G.nodes[node].get("heading", node)
        node_text.append(f"{node}<br>{heading[:40]}")

        sev = severity_map.get(node, None)
        if sev == "High":
            node_color.append("#e53935")
            node_size.append(22)
        elif sev == "Medium":
            node_color.append("#fb8c00")
            node_size.append(18)
        elif sev == "Low":
            node_color.append("#fdd835")
            node_size.append(15)
        else:
            node_color.append("#90caf9")
            node_size.append(12)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n.split("<br>")[0] for n in node_text],
        hovertext=node_text,
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color="#333")
        ),
        showlegend=False
    )

    # Legend items
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#e53935"),
                   name="High severity"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#fb8c00"),
                   name="Medium severity"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#fdd835"),
                   name="Low severity"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#90caf9"),
                   name="No contradiction"),
    ]

    fig = go.Figure(
        data=edge_traces + [node_trace] + legend_traces,
        layout=go.Layout(
            title="Contradiction Network Graph",
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=520,
            margin=dict(t=60, b=20, l=20, r=20)
        )
    )
    return fig


def make_section_risk_bar(contradictions: list, chunks: list) -> go.Figure:
    """Horizontal bar showing risk score per section."""
    risk = {}
    for c in contradictions:
        score = {"High": 3, "Medium": 2, "Low": 1}.get(c.get("severity", "Low"), 1)
        score = score * (c.get("confidence", 75) / 100)
        risk[c["target"]] = risk.get(c["target"], 0) + score

    if not risk:
        return go.Figure()

    sorted_risk = sorted(risk.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_risk]
    scores = [round(x[1], 2) for x in sorted_risk]

    colors = []
    for s in scores:
        if s >= 2.5:
            colors.append("#e53935")
        elif s >= 1.5:
            colors.append("#fb8c00")
        else:
            colors.append("#fdd835")

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=scores,
        textposition="outside"
    ))
    fig.update_layout(
        title="Section Risk Score (severity × confidence)",
        xaxis_title="Risk Score",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(labels) * 40 + 80),
        margin=dict(t=50, b=40, l=130, r=60)
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL-TEXTUAL CHARTS (Novel)
# ─────────────────────────────────────────────────────────────────────────────

def make_visual_contradiction_summary(visual_contradictions: list) -> go.Figure:
    """
    Bar chart breaking down visual-textual contradiction types.
    Shows the distribution across Numerical, Trend, Scope, etc.
    """
    type_counts = {}
    for vc in visual_contradictions:
        t = vc.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    if not type_counts:
        return go.Figure()

    labels = list(type_counts.keys())
    values = list(type_counts.values())

    # Color mapping for visual types
    color_map = {
        "Visual-Numerical":   "#e53935",
        "Visual-Trend":       "#fb8c00",
        "Visual-Scope":       "#fdd835",
        "Visual-Categorical": "#66bb6a",
        "Visual-Temporal":    "#42a5f5",
        "Visual-Omission":    "#ab47bc",
    }
    colors = [color_map.get(l, "#8888ff") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside"
    ))
    fig.update_layout(
        title="Visual-Textual Contradiction Types",
        xaxis_title="Contradiction Type",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddddf5"),
        height=350,
        margin=dict(t=50, b=80, l=40, r=20),
        xaxis=dict(tickangle=-30),
    )
    return fig


def make_cross_modal_network(
    text_chunks: list,
    chart_chunks: list,
    visual_contradictions: list,
) -> go.Figure:
    """
    Network graph showing cross-modal relationships.
    Purple nodes = chart chunks, Blue nodes = text sections.
    Edges connect charts to text sections they contradict.
    """
    G = nx.Graph()

    # Add text nodes
    for chunk in text_chunks:
        G.add_node(chunk["label"], node_type="text",
                   heading=chunk.get("heading", chunk["label"]))

    # Add chart nodes
    for chunk in chart_chunks:
        G.add_node(chunk["label"], node_type="chart",
                   heading=chunk.get("heading", chunk["label"]))

    # Add edges from visual contradictions
    for vc in visual_contradictions:
        target = vc["target"]
        conf = vc.get("confidence", 75)
        sev = vc.get("severity", "Low")
        for related in vc.get("related_sections", []):
            if G.has_node(target) and G.has_node(related):
                G.add_edge(target, related, weight=conf, severity=sev)

    if len(G.nodes) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Edges
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 50)
        sev = data.get("severity", "Low")
        color = {"High": "#e53935", "Medium": "#fb8c00", "Low": "#fdd835"}.get(sev, "#aaa")
        width = max(1, weight / 25)

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none", showlegend=False,
        ))

    # Nodes
    node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = G.nodes[node].get("node_type", "text")
        heading = G.nodes[node].get("heading", node)
        node_text.append(f"{node}<br>{heading[:40]}")

        if ntype == "chart":
            node_color.append("#aa66ff")
            node_size.append(20)
            node_symbol.append("diamond")
        else:
            node_color.append("#4488ff")
            node_size.append(14)
            node_symbol.append("circle")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n.split("<br>")[0] for n in node_text],
        hovertext=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#ddddf5"),
        marker=dict(
            color=node_color, size=node_size, symbol=node_symbol,
            line=dict(width=1.5, color="#333"),
        ),
        showlegend=False,
    )

    # Legend
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#aa66ff", symbol="diamond"),
                   name="Chart / Graph"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#4488ff"),
                   name="Text Section"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#e53935"),
                   name="High severity edge"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color="#fb8c00"),
                   name="Medium severity edge"),
    ]

    fig = go.Figure(
        data=edge_traces + [node_trace] + legend_traces,
        layout=go.Layout(
            title="Cross-Modal Contradiction Network",
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ddddf5"),
            height=520,
            margin=dict(t=60, b=20, l=20, r=20),
        )
    )
    return fig