"""
neo4j_graph.py — compatibility shim
===================================

All graph functionality has been consolidated into ``graph_store.py``.
This module is kept so that existing imports (``from pipeline.neo4j_graph
import get_graph, NEO4J_AVAILABLE``) continue to work.

Everything exported here is re-exported from ``graph_store``.  No
behaviour is defined in this file; edit ``graph_store.py`` instead.
"""

from pipeline.graph_store import (  # noqa: F401
    NEO4J_AVAILABLE,
    ContradictionGraph,
    get_graph,
    get_driver,
    is_available,
    close,
    # functional API (handy for callers that want either style)
    build_graph,
    clear_graph,
    extract_entities,
    get_transitive_contradictions,
    get_entity_contradiction_clusters,
    get_most_contradictory_sections,
    get_graph_stats,
    ENTITY_PATTERNS,
)

__all__ = [
    "NEO4J_AVAILABLE", "ContradictionGraph", "get_graph",
    "get_driver", "is_available", "close",
    "build_graph", "clear_graph", "extract_entities",
    "get_transitive_contradictions",
    "get_entity_contradiction_clusters",
    "get_most_contradictory_sections",
    "get_graph_stats",
    "ENTITY_PATTERNS",
]
