"""
Contradiction Knowledge Graph — Unified Neo4j Integration
=========================================================

This module is the single source of truth for all Neo4j-related graph
functionality in ContradictAI.  It consolidates the former split between
``graph_store.py`` (functional API, used by app.py) and
``neo4j_graph.py`` (class-based API, used by graph_tab_integration.py).

It exposes:

1. A **functional API** (``build_graph``, ``get_transitive_contradictions``
   etc.) used directly by ``app.py``.
2. A **class-based API** (``ContradictionGraph`` + the ``get_graph()``
   factory) used by ``graph_tab_integration.py`` for the rich "Graph
   Intelligence" tab.

Both APIs operate on the *same* underlying schema, so results are
consistent regardless of which entry point is used.

SCHEMA
------
    (Document {name, doc_name})
        -[:HAS_SECTION]->
    (Section {label, heading, display, text_preview, chunk_id, doc_name})
        -[:CONTAINS]->
    (Statement {id, text, section_label, doc_name})
        -[:MENTIONS]->
    (Entity {name, doc_name})

    (Statement)-[:CONTRADICTS {score, type, severity, confidence}]->(Statement)
    (Section)-[:CONTRADICTS_SECTION {count, max_score, types}]->(Section)
    (Section)-[:CONTRADICTS {confidence, severity, type}]->(Section)
          ↑ legacy edge produced by ``ContradictionGraph.store_contradiction``
            used by the cycle / propagation queries

QUERIES EXPOSED
---------------
    * Transitive contradictions     (A→B→C where A-C not directly linked)
    * Contradiction chains          (paths of length 2..N)
    * Contradiction loops           (circular chains A→...→A)
    * Hotspot sections              (degree centrality)
    * Entity-centric clusters       (topics with most conflicts)
    * Transitive risk / propagation score (novel decayed metric)
    * Graph statistics              (counts for UI cards)
    * Cross-document comparison     (A in doc1 ↔ B in doc2)

NEO4J AVAILABILITY
------------------
Graceful degradation: if the ``neo4j`` package isn't installed or the
server is unreachable, every function returns an empty / None result and
logs a warning.  Callers should treat graph output as optional.

CONNECTION
----------
Connection parameters are read from env (``.env`` via python-dotenv):

    NEO4J_URI       (default: bolt://localhost:7687)
    NEO4J_USER      or  NEO4J_USERNAME   (default: neo4j)
    NEO4J_PASSWORD  (default: password)

Both ``NEO4J_USER`` and ``NEO4J_USERNAME`` are accepted so existing
``.env`` files from either previous module keep working.
"""

from __future__ import annotations

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION (shared singleton driver)
# ─────────────────────────────────────────────────────────────────────────────

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_driver = None


def get_driver():
    """Get or create the shared Neo4j driver. Returns None if unavailable."""
    global _driver
    if not NEO4J_AVAILABLE:
        return None
    if _driver is not None:
        return _driver
    try:
        _driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        _driver.verify_connectivity()
        _ensure_indexes(_driver)
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        return _driver
    except Exception as e:
        logger.warning(f"Neo4j not available ({e}). Graph features disabled.")
        _driver = None
        return None


def is_available() -> bool:
    """Public check used by UI code."""
    return get_driver() is not None


def close():
    """Close the shared driver. Safe to call multiple times."""
    global _driver
    if _driver:
        try:
            _driver.close()
        except Exception:
            pass
        _driver = None


def _ensure_indexes(driver):
    """Create indexes once per connection so queries stay fast."""
    try:
        with driver.session() as s:
            s.run("CREATE INDEX section_doc IF NOT EXISTS "
                  "FOR (n:Section) ON (n.doc_name)")
            s.run("CREATE INDEX section_label IF NOT EXISTS "
                  "FOR (n:Section) ON (n.label)")
            s.run("CREATE INDEX statement_doc IF NOT EXISTS "
                  "FOR (n:Statement) ON (n.doc_name)")
            s.run("CREATE INDEX entity_doc IF NOT EXISTS "
                  "FOR (n:Entity) ON (n.doc_name)")
    except Exception as e:
        logger.debug(f"Index creation skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION (lightweight regex, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_PATTERNS = [
    # Security
    (r'\b(MFA|multi-factor authentication)\b', 'MFA'),
    (r'\b(VPN|virtual private network)\b', 'VPN'),
    (r'\b(password[s]?)\b', 'Password'),
    (r'\b(encryption|encrypted)\b', 'Encryption'),
    (r'\b(security audit[s]?|audit[s]?)\b', 'Audit'),
    (r'\b(firewall[s]?)\b', 'Firewall'),
    # Work policy
    (r'\b(remote work|work remotely|working remotely)\b', 'Remote Work'),
    (r'\b(work(?:ing)? hours|core hours)\b', 'Work Hours'),
    (r'\b(flexible scheduling|flexible hours)\b', 'Flexible Scheduling'),
    (r'\b(leave|time[- ]off)\b', 'Leave Policy'),
    (r'\b(monitoring|screen tracking)\b', 'Monitoring'),
    (r'\b(compliance|comply)\b', 'Compliance'),
    # Data
    (r'\b(sensitive data|confidential)\b', 'Sensitive Data'),
    (r'\b(local storage|stored locally)\b', 'Local Storage'),
    (r'\b(cloud|centralized)\b', 'Cloud Storage'),
    (r'\b(personal device[s]?|BYOD)\b', 'Personal Devices'),
    (r'\b(company[- ]issued device[s]?)\b', 'Company Devices'),
    # Communication
    (r'\b(email)\b', 'Email'),
    (r'\b(instant messaging|messaging apps)\b', 'Messaging'),
    (r'\b(public Wi-?Fi)\b', 'Public WiFi'),
    # People
    (r'\b(employee[s]?)\b', 'Employees'),
    (r'\b(manager[s]?)\b', 'Managers'),
    (r'\b(intern[s]?)\b', 'Interns'),
    (r'\b(contractor?[s]?|contract employee[s]?)\b', 'Contractors'),
]


def extract_entities(text: str) -> list[str]:
    """Extract topic entities from text via regex pattern matching."""
    entities = set()
    text_lower = text.lower()
    for pattern, name in ENTITY_PATTERNS:
        if re.search(pattern, text_lower):
            entities.add(name)
    return sorted(entities)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONAL API — used by app.py
# ─────────────────────────────────────────────────────────────────────────────

def clear_graph(doc_name: str):
    """Delete every node / edge belonging to ``doc_name``."""
    driver = get_driver()
    if not driver:
        return
    with driver.session() as s:
        s.run(
            "MATCH (n {doc_name: $doc}) DETACH DELETE n",
            doc=doc_name,
        )


def build_graph(
    doc_name: str,
    chunks: list[dict],
    contradictions: list[dict],
) -> None:
    """
    Build the full contradiction knowledge graph for ``doc_name``.

    Creates Document/Section/Statement/Entity nodes plus CONTRADICTS
    edges at both the statement and section level.  Idempotent on
    re-upload: the document is wiped first.
    """
    driver = get_driver()
    if not driver:
        return

    clear_graph(doc_name)

    with driver.session() as session:
        # Document node
        session.run(
            """
            CREATE (d:Document {
                name: $name, doc_name: $doc,
                sections: $n_sections, contradictions: $n_contradictions
            })
            """,
            name=doc_name,
            doc=doc_name,
            n_sections=len(chunks),
            n_contradictions=len(contradictions),
        )

        # Section nodes + Statement + Entity
        for chunk in chunks:
            session.run(
                """
                MATCH (d:Document {doc_name: $doc})
                CREATE (s:Section {
                    label:        $label,
                    display:      $display,
                    heading:      $heading,
                    text_preview: $preview,
                    chunk_id:     $id,
                    char_count:   $chars,
                    doc_name:     $doc
                })
                CREATE (d)-[:HAS_SECTION]->(s)
                """,
                doc=doc_name,
                label=chunk["label"],
                display=chunk.get("display", chunk["label"]),
                heading=chunk.get("heading", ""),
                preview=chunk["text"][:300],
                id=chunk.get("id", 0),
                chars=len(chunk["text"]),
            )

            # Split into statements using the same helper as the rest of
            # the pipeline — keeps node text aligned with what the NLI
            # stage saw.
            try:
                from pipeline.nli_filter import split_into_statements
            except Exception:
                from .nli_filter import split_into_statements  # fallback

            statements = split_into_statements(chunk["text"])
            for j, stmt in enumerate(statements):
                stmt_id = f"{chunk['label']}_stmt_{j}"
                session.run(
                    """
                    MATCH (s:Section {label: $label, doc_name: $doc})
                    CREATE (st:Statement {
                        id:            $stmt_id,
                        text:          $text,
                        section_label: $label,
                        doc_name:      $doc
                    })
                    CREATE (s)-[:CONTAINS]->(st)
                    """,
                    doc=doc_name,
                    label=chunk["label"],
                    stmt_id=stmt_id,
                    text=stmt,
                )

                for entity in extract_entities(stmt):
                    session.run(
                        """
                        MERGE (e:Entity {name: $name, doc_name: $doc})
                        WITH e
                        MATCH (st:Statement {id: $stmt_id, doc_name: $doc})
                        MERGE (st)-[:MENTIONS]->(e)
                        """,
                        doc=doc_name,
                        name=entity,
                        stmt_id=stmt_id,
                    )

        # Contradiction edges — statement-level when quotes parse out,
        # section-level aggregated either way.
        for c in contradictions:
            analysis = c.get("analysis", "")
            stmt_a_match = re.search(r'Statement A[^"]*"([^"]+)"', analysis)
            stmt_b_match = re.search(r'Statement B[^"]*"([^"]+)"', analysis)

            if stmt_a_match and stmt_b_match:
                stmt_a_text = stmt_a_match.group(1)[:200]
                stmt_b_text = stmt_b_match.group(1)[:200]

                session.run(
                    """
                    MATCH (a:Statement {doc_name: $doc})
                    WHERE a.text CONTAINS $stmt_a
                    WITH a LIMIT 1
                    MATCH (b:Statement {doc_name: $doc})
                    WHERE b.text CONTAINS $stmt_b
                    WITH a, b LIMIT 1
                    CREATE (a)-[:CONTRADICTS {
                        score:      $score,
                        type:       $type,
                        severity:   $severity,
                        confidence: $confidence
                    }]->(b)
                    """,
                    doc=doc_name,
                    stmt_a=stmt_a_text[:80],
                    stmt_b=stmt_b_text[:80],
                    score=c.get("confidence", 0) / 100.0,
                    type=c.get("type", "Direct"),
                    severity=c.get("severity", "Medium"),
                    confidence=c.get("confidence", 0),
                )

            # Section-level aggregated edges (CONTRADICTS_SECTION) +
            # simple Section→Section CONTRADICTS edge for cycle queries.
            target_label = c.get("target", "")
            for related in c.get("related_sections", []):
                related_clean = related.replace(" (same section)", "")

                # Aggregated edge with counts
                session.run(
                    """
                    MATCH (a:Section {label: $label_a, doc_name: $doc})
                    MATCH (b:Section {label: $label_b, doc_name: $doc})
                    MERGE (a)-[r:CONTRADICTS_SECTION]->(b)
                    ON CREATE SET r.count = 1, r.max_score = $score, r.types = [$type]
                    ON MATCH  SET r.count = r.count + 1,
                                  r.max_score = CASE WHEN $score > r.max_score
                                                 THEN $score ELSE r.max_score END,
                                  r.types = r.types + $type
                    """,
                    doc=doc_name,
                    label_a=target_label,
                    label_b=related_clean,
                    score=c.get("confidence", 0) / 100.0,
                    type=c.get("type", "Direct"),
                )

                # Simple Section→Section CONTRADICTS edge used by cycle /
                # propagation-score queries (formerly in neo4j_graph.py).
                session.run(
                    """
                    MATCH (a:Section {label: $label_a, doc_name: $doc})
                    MATCH (b:Section {label: $label_b, doc_name: $doc})
                    MERGE (a)-[r:CONTRADICTS]->(b)
                    SET r.confidence = $conf,
                        r.severity   = $sev,
                        r.type       = $ctype
                    """,
                    doc=doc_name,
                    label_a=target_label,
                    label_b=related_clean,
                    conf=c.get("confidence", 0),
                    sev=c.get("severity", "Medium"),
                    ctype=c.get("type", "Direct"),
                )

    logger.info(
        f"Built knowledge graph for '{doc_name}': "
        f"{len(chunks)} sections, {len(contradictions)} contradictions"
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUERIES — functional API
# ─────────────────────────────────────────────────────────────────────────────

def get_transitive_contradictions(doc_name: str) -> list[dict]:
    """
    Statement-level transitive contradictions: A → B → C where A and C
    are not directly linked.  Core multi-hop insight used by app.py.
    """
    driver = get_driver()
    if not driver:
        return []
    with driver.session() as s:
        result = s.run(
            """
            MATCH (a:Statement {doc_name: $doc})-[r1:CONTRADICTS]->(b:Statement)
                  -[r2:CONTRADICTS]->(c:Statement)
            WHERE a <> c
              AND NOT (a)-[:CONTRADICTS]->(c)
              AND NOT (c)-[:CONTRADICTS]->(a)
            RETURN a.text AS stmt_a, a.section_label AS section_a,
                   b.text AS bridge, b.section_label AS section_bridge,
                   c.text AS stmt_c, c.section_label AS section_c,
                   r1.score AS score_ab, r2.score AS score_bc
            """,
            doc=doc_name,
        )
        return [dict(record) for record in result]


def get_entity_contradiction_clusters(doc_name: str) -> list[dict]:
    """Group contradictions by the entities they mention (topic hotspots)."""
    driver = get_driver()
    if not driver:
        return []
    with driver.session() as s:
        result = s.run(
            """
            MATCH (a:Statement {doc_name: $doc})-[:CONTRADICTS]->(b:Statement)
            MATCH (a)-[:MENTIONS]->(e:Entity)
            WITH e.name AS entity,
                 count(*) AS contradiction_count,
                 collect(DISTINCT a.text)[..3] AS sample_statements
            WHERE contradiction_count > 0
            RETURN entity, contradiction_count, sample_statements
            ORDER BY contradiction_count DESC
            """,
            doc=doc_name,
        )
        return [dict(record) for record in result]


def get_most_contradictory_sections(doc_name: str) -> list[dict]:
    """Rank sections by statement-level contradiction centrality."""
    driver = get_driver()
    if not driver:
        return []
    with driver.session() as s:
        result = s.run(
            """
            MATCH (s:Section {doc_name: $doc})
            OPTIONAL MATCH (s)-[:CONTAINS]->(st:Statement)-[r:CONTRADICTS]-()
            WITH s.label AS section, s.display AS display,
                 count(r) AS contradiction_edges
            RETURN section, display, contradiction_edges
            ORDER BY contradiction_edges DESC
            """,
            doc=doc_name,
        )
        return [dict(record) for record in result]


def get_graph_stats(doc_name: str) -> dict:
    """Summary counts used by the sidebar metrics cards."""
    driver = get_driver()
    if not driver:
        return {}
    with driver.session() as s:
        result = s.run(
            """
            MATCH (s:Section {doc_name: $doc})
            WITH count(s) AS sections
            OPTIONAL MATCH (st:Statement {doc_name: $doc})
            WITH sections, count(st) AS statements
            OPTIONAL MATCH (e:Entity {doc_name: $doc})
            WITH sections, statements, count(e) AS entities
            OPTIONAL MATCH (:Statement {doc_name: $doc})-[r:CONTRADICTS]->()
            WITH sections, statements, entities, count(r) AS contradiction_edges
            OPTIONAL MATCH (:Statement {doc_name: $doc})-[m:MENTIONS]->()
            RETURN sections, statements, entities,
                   contradiction_edges, count(m) AS mention_edges
            """,
            doc=doc_name,
        )
        record = result.single()
        return dict(record) if record else {}


# ─────────────────────────────────────────────────────────────────────────────
# CLASS-BASED API — used by graph_tab_integration.render_graph_tab
# ─────────────────────────────────────────────────────────────────────────────

class ContradictionGraph:
    """
    OO wrapper around the shared driver.  Exposes the richer queries
    (chains, loops, propagation, cross-doc) on top of the section-level
    CONTRADICTS edges produced by ``build_graph``.

    Backward compatible with the old ``neo4j_graph.ContradictionGraph``:
    every public method name and signature is preserved.
    """

    def __init__(self, uri=None, username=None, password=None):
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package not installed. Run: pip install neo4j"
            )
        # Allow per-instance overrides; default to the shared driver so
        # we don't open a second connection in the common case.
        if uri or username or password:
            self.driver = GraphDatabase.driver(
                uri or NEO4J_URI,
                auth=(username or NEO4J_USER, password or NEO4J_PASSWORD),
            )
            _ensure_indexes(self.driver)
            self._owns_driver = True
        else:
            shared = get_driver()
            if shared is None:
                raise ConnectionError(
                    f"Cannot reach Neo4j at {NEO4J_URI}"
                )
            self.driver = shared
            self._owns_driver = False

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self):
        """Close only if this instance owns the driver."""
        if self._owns_driver and self.driver:
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = None

    @staticmethod
    def is_available() -> bool:
        return NEO4J_AVAILABLE

    # ── writes (section-only schema, idempotent) ────────────────────

    def clear_document(self, doc_name: str):
        with self.driver.session() as s:
            s.run(
                "MATCH (n {doc_name: $doc}) DETACH DELETE n",
                doc=doc_name,
            )

    def store_sections(self, doc_name: str, chunks: list):
        with self.driver.session() as s:
            for chunk in chunks:
                s.run(
                    """
                    MERGE (n:Section {doc_name: $doc, label: $label})
                    SET n.heading      = $heading,
                        n.text_preview = $preview,
                        n.chunk_id     = $id,
                        n.display      = $display
                    """,
                    doc=doc_name,
                    label=chunk["label"],
                    heading=chunk.get("heading", ""),
                    preview=chunk["text"][:300],
                    id=chunk.get("id", 0),
                    display=chunk.get("display", chunk["label"]),
                )

    def store_contradiction(self, doc_name: str, contradiction: dict):
        target = contradiction["target"]
        confidence = contradiction.get("confidence", 70)
        severity = contradiction.get("severity", "Low")
        c_type = contradiction.get("type", "Direct")

        with self.driver.session() as s:
            for related in contradiction.get("related_sections", []):
                related_clean = related.replace(" (same section)", "")
                s.run(
                    """
                    MATCH (a:Section {doc_name: $doc, label: $target})
                    MATCH (b:Section {doc_name: $doc, label: $related})
                    MERGE (a)-[r:CONTRADICTS]->(b)
                    SET r.confidence = $conf,
                        r.severity   = $sev,
                        r.type       = $ctype
                    """,
                    doc=doc_name,
                    target=target,
                    related=related_clean,
                    conf=confidence,
                    sev=severity,
                    ctype=c_type,
                )

    def store_all(self, doc_name: str, chunks: list, contradictions: list):
        """
        Convenience wrapper: calls the richer functional ``build_graph``
        so the full Statement / Entity schema is populated as well.
        """
        build_graph(doc_name, chunks, contradictions)

    # ── rich graph queries ──────────────────────────────────────────

    def get_contradiction_chains(
        self,
        doc_name: str,
        min_hops: int = 2,
        max_hops: int = 5,
    ) -> list[dict]:
        """Paths A → B → C ... of length 2..N between distinct sections."""
        query = f"""
        MATCH path = (a:Section {{doc_name: $doc}})
          -[:CONTRADICTS*{min_hops}..{max_hops}]->
          (b:Section {{doc_name: $doc}})
        WHERE a <> b
        WITH [n IN nodes(path) | n.label]  AS chain,
             length(path)                  AS depth,
             [r IN relationships(path) | r.confidence] AS confs
        RETURN chain, depth,
               reduce(m = 100, c IN confs |
                      CASE WHEN c < m THEN c ELSE m END) AS min_confidence
        ORDER BY depth DESC, min_confidence DESC
        LIMIT 30
        """
        with self.driver.session() as s:
            result = s.run(query, doc=doc_name)
            return [
                {
                    "chain": r["chain"],
                    "depth": r["depth"],
                    "min_confidence": r["min_confidence"],
                }
                for r in result
            ]

    def get_contradiction_loops(self, doc_name: str) -> list[dict]:
        """Circular contradictions A → ... → A."""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH path = (a:Section {doc_name: $doc})
                  -[:CONTRADICTS*2..6]->(a)
                WITH [n IN nodes(path) | n.label] AS loop,
                     length(path) AS size
                RETURN DISTINCT loop, size
                ORDER BY size
                LIMIT 10
                """,
                doc=doc_name,
            )
            return [{"loop": r["loop"], "size": r["size"]} for r in result]

    def get_hotspot_sections(
        self,
        doc_name: str,
        top_n: int = 10,
    ) -> list[dict]:
        """Sections with the highest contradiction degree."""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (s:Section {doc_name: $doc})
                      -[r:CONTRADICTS]-(other:Section)
                RETURN s.label           AS section,
                       s.heading         AS heading,
                       count(other)      AS degree,
                       collect(DISTINCT other.label) AS conflicts_with,
                       avg(r.confidence) AS avg_confidence
                ORDER BY degree DESC
                LIMIT $n
                """,
                doc=doc_name, n=top_n,
            )
            return [dict(r) for r in result]

    def get_transitive_risk(
        self,
        section_label: str,
        doc_name: str,
    ) -> list[dict]:
        """
        Novel propagation metric: every section reachable from
        ``section_label`` with a score that decays 0.7^(hops-1).
        """
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH path = (start:Section {doc_name: $doc, label: $label})
                  -[:CONTRADICTS*1..4]->(reached:Section {doc_name: $doc})
                WHERE reached <> start
                WITH reached.label AS section,
                     length(path)  AS hops,
                     [r IN relationships(path) | r.confidence] AS confs,
                     [n IN nodes(path) | n.label] AS path_labels
                WITH section, hops, path_labels,
                     reduce(p = 1.0, c IN confs | p * (c / 100.0)) AS raw_score
                RETURN section,
                       hops,
                       round(raw_score * (0.7 ^ (hops - 1)) * 100)
                       AS propagation_score,
                       path_labels
                ORDER BY propagation_score DESC
                """,
                doc=doc_name, label=section_label,
            )
            return [dict(r) for r in result]

    def get_graph_summary(self, doc_name: str) -> dict:
        """Quick stats for the graph-tab header (sections / edges / avg conf)."""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (s:Section {doc_name: $doc})
                OPTIONAL MATCH (s)-[r:CONTRADICTS]-()
                RETURN count(DISTINCT s)   AS sections,
                       count(DISTINCT r)   AS edges,
                       avg(r.confidence)   AS avg_confidence
                """,
                doc=doc_name,
            )
            row = result.single()
            return {
                "sections":       row["sections"],
                "edges":          row["edges"],
                "avg_confidence": round(row["avg_confidence"] or 0, 1),
            }

    def compare_documents(self, doc1: str, doc2: str) -> list[dict]:
        """Cross-document CONTRADICTS edges (doc1 ↔ doc2)."""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (a:Section {doc_name: $doc1})
                      -[r:CONTRADICTS]->
                      (b:Section {doc_name: $doc2})
                RETURN a.label AS section_doc1,
                       b.label AS section_doc2,
                       r.severity   AS severity,
                       r.confidence AS confidence,
                       r.type       AS type
                ORDER BY r.confidence DESC
                """,
                doc1=doc1, doc2=doc2,
            )
            return [dict(r) for r in result]

    def list_documents(self) -> list[str]:
        """All distinct ``doc_name`` values in the graph."""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (n:Section)
                RETURN DISTINCT n.doc_name AS doc
                ORDER BY doc
                """
            )
            return [r["doc"] for r in result if r["doc"]]

    # ── interactive-viz data export ─────────────────────────────────
    def get_graph_nodes_and_edges(self, doc_name: str) -> dict:
        """
        Fetch section nodes + CONTRADICTS edges for interactive rendering.

        Returns a dict:
            {
              "nodes": [{id, label, heading, degree, preview}, ...],
              "edges": [{source, target, confidence, severity, type}, ...],
              "severity_counts": {"High": n, "Medium": n, "Low": n}
            }

        Uses the section-level CONTRADICTS edge (set in build_graph and in
        store_contradiction) so a single pass populates the whole view.
        """
        with self.driver.session() as s:
            # Nodes: every Section in this doc, plus its edge degree
            node_result = s.run(
                """
                MATCH (s:Section {doc_name: $doc})
                OPTIONAL MATCH (s)-[r:CONTRADICTS]-()
                RETURN s.label           AS id,
                       coalesce(s.display, s.label) AS label,
                       coalesce(s.heading, "")     AS heading,
                       coalesce(s.text_preview, "") AS preview,
                       count(DISTINCT r) AS degree
                """,
                doc=doc_name,
            )
            nodes = [dict(r) for r in node_result]

            # Edges: directed Section→Section CONTRADICTS (we render
            # undirected but keep the confidence/severity/type props).
            edge_result = s.run(
                """
                MATCH (a:Section {doc_name: $doc})
                      -[r:CONTRADICTS]->
                      (b:Section {doc_name: $doc})
                RETURN a.label AS source,
                       b.label AS target,
                       coalesce(r.confidence, 0)  AS confidence,
                       coalesce(r.severity, "Low") AS severity,
                       coalesce(r.type, "Direct")  AS type
                """,
                doc=doc_name,
            )
            edges_raw = [dict(r) for r in edge_result]

        # De-duplicate undirected pairs, keep the highest-confidence entry
        seen_pairs: dict[tuple[str, str], dict] = {}
        for e in edges_raw:
            key = tuple(sorted((e["source"], e["target"])))
            prev = seen_pairs.get(key)
            if prev is None or e["confidence"] > prev["confidence"]:
                seen_pairs[key] = e
        edges = list(seen_pairs.values())

        sev_counts = {"High": 0, "Medium": 0, "Low": 0}
        for e in edges:
            sev_counts[e["severity"]] = sev_counts.get(e["severity"], 0) + 1

        return {"nodes": nodes, "edges": edges, "severity_counts": sev_counts}


def get_graph(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[ContradictionGraph]:
    """
    Safe factory: return a ``ContradictionGraph`` if Neo4j is reachable,
    otherwise ``None``.  Use this from the UI so missing Neo4j degrades
    gracefully.
    """
    if not NEO4J_AVAILABLE:
        return None
    try:
        return ContradictionGraph(uri, username, password)
    except Exception as e:
        logger.warning(f"get_graph() failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # connection
    "get_driver", "is_available", "close", "NEO4J_AVAILABLE",
    # entity helpers
    "ENTITY_PATTERNS", "extract_entities",
    # functional API
    "clear_graph", "build_graph",
    "get_transitive_contradictions",
    "get_entity_contradiction_clusters",
    "get_most_contradictory_sections",
    "get_graph_stats",
    # class API
    "ContradictionGraph", "get_graph",
]
