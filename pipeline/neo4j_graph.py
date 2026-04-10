"""
Neo4j Contradiction Graph
==========================

WHY NEO4J OVER A FLAT LIST
---------------------------
The existing pipeline stores contradictions as a flat Python list.
That lets you count them, filter by severity, and show a heatmap.

But it cannot answer:
  - "Does Section 3 *transitively* conflict with Section 17 via an
     intermediate section?" (contradiction chains)
  - "Are there circular contradictions — A contradicts B, B contradicts C,
     C contradicts A?" (loops — impossible in a list)
  - "Which section is the single biggest source of instability across the
     entire document?" (PageRank-style centrality)
  - "Compare two versions of the same policy document" (cross-doc edges)

Neo4j gives all of this for free via Cypher queries.

GRAPH SCHEMA
-------------
  (:Section {doc, label, heading, text_preview, id})
      -[:CONTRADICTS {confidence, severity, type}]->
  (:Section {doc, label, ...})

SETUP (free options)
---------------------
  Local:   install Neo4j Desktop → create DB → bolt://localhost:7687
  Cloud:   https://neo4j.com/cloud/platform/aura-graph-database/
           (AuraDB Free — no credit card, 200k nodes)

Add to .env:
  NEO4J_URI=bolt://localhost:7687   or  neo4j+s://xxxxx.databases.neo4j.io
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=yourpassword
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class ContradictionGraph:
    """
    Stores sections and their contradiction relationships in Neo4j,
    then provides graph-native queries that a flat list cannot answer.
    """

    def __init__(self, uri=None, username=None, password=None):
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package not installed. Run: pip install neo4j"
            )
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._ensure_indexes()

    def close(self):
        self.driver.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_indexes(self):
        """Create indexes on first use so queries stay fast as data grows."""
        with self.driver.session() as s:
            s.run("CREATE INDEX section_doc IF NOT EXISTS "
                  "FOR (n:Section) ON (n.doc)")
            s.run("CREATE INDEX section_label IF NOT EXISTS "
                  "FOR (n:Section) ON (n.label)")

    # ── Write ─────────────────────────────────────────────────────────────────

    def clear_document(self, doc_name: str):
        """Remove all nodes and edges for a document (e.g. on re-upload)."""
        with self.driver.session() as s:
            s.run(
                "MATCH (n:Section {doc: $doc}) DETACH DELETE n",
                doc=doc_name
            )

    def store_sections(self, doc_name: str, chunks: list):
        """
        Create a :Section node for every chunk.
        Uses MERGE so re-running is idempotent.
        """
        with self.driver.session() as s:
            for chunk in chunks:
                s.run(
                    """
                    MERGE (n:Section {doc: $doc, label: $label})
                    SET n.heading      = $heading,
                        n.text_preview = $preview,
                        n.chunk_id     = $id
                    """,
                    doc=doc_name,
                    label=chunk["label"],
                    heading=chunk.get("heading", ""),
                    preview=chunk["text"][:300],
                    id=chunk["id"]
                )

    def store_contradiction(self, doc_name: str, contradiction: dict):
        """
        Create a :CONTRADICTS edge between the target section and every
        related section it was found to conflict with.
        Edge carries confidence, severity, and contradiction type.
        """
        target = contradiction["target"]
        confidence = contradiction.get("confidence", 70)
        severity = contradiction.get("severity", "Low")
        c_type = contradiction.get("type", "Direct")

        with self.driver.session() as s:
            for related in contradiction.get("related_sections", []):
                s.run(
                    """
                    MATCH (a:Section {doc: $doc, label: $target})
                    MATCH (b:Section {doc: $doc, label: $related})
                    MERGE (a)-[r:CONTRADICTS]->(b)
                    SET r.confidence = $conf,
                        r.severity   = $sev,
                        r.type       = $ctype
                    """,
                    doc=doc_name,
                    target=target,
                    related=related,
                    conf=confidence,
                    sev=severity,
                    ctype=c_type
                )

    def store_all(self, doc_name: str, chunks: list, contradictions: list):
        """Convenience: store sections then all contradictions."""
        self.store_sections(doc_name, chunks)
        for c in contradictions:
            self.store_contradiction(doc_name, c)

    # ── Unique graph queries ──────────────────────────────────────────────────

    def get_contradiction_chains(self, doc_name: str,
                                 min_hops: int = 2,
                                 max_hops: int = 5) -> list:
        """
        Find paths where Section A contradicts B which contradicts C ...
        These are impossible to detect with a flat list — they require
        graph traversal across multiple edges.

        Returns list of:
          {chain: ["Section 1", "Section 4", "Section 9"],
           depth: 3,
           min_confidence: 72}
        """
        query = f"""
        MATCH path = (a:Section {{doc: $doc}})
          -[:CONTRADICTS*{min_hops}..{max_hops}]->
          (b:Section {{doc: $doc}})
        WHERE a <> b
          AND all(n IN nodes(path) WHERE n.doc = $doc)
        WITH [n IN nodes(path) | n.label]  AS chain,
             length(path)                  AS depth,
             [r IN relationships(path) | r.confidence] AS confs
        RETURN chain, depth,
               reduce(m = 100, c IN confs | CASE WHEN c < m THEN c ELSE m END)
               AS min_confidence
        ORDER BY depth DESC, min_confidence DESC
        LIMIT 30
        """
        with self.driver.session() as s:
            result = s.run(query, doc=doc_name)
            return [
                {
                    "chain": r["chain"],
                    "depth": r["depth"],
                    "min_confidence": r["min_confidence"]
                }
                for r in result
            ]

    def get_contradiction_loops(self, doc_name: str) -> list:
        """
        Find circular contradiction chains: A → B → ... → A.
        These are the most dangerous findings — they mean the document
        contains a ring of inconsistencies with no consistent resolution.

        Returns list of:
          {loop: ["Section 2", "Section 7", "Section 2"], size: 2}
        """
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH path = (a:Section {doc: $doc})
                  -[:CONTRADICTS*2..6]->(a)
                WITH [n IN nodes(path) | n.label] AS loop,
                     length(path) AS size
                RETURN DISTINCT loop, size
                ORDER BY size
                LIMIT 10
                """,
                doc=doc_name
            )
            return [{"loop": r["loop"], "size": r["size"]} for r in result]

    def get_hotspot_sections(self, doc_name: str, top_n: int = 10) -> list:
        """
        Sections with the highest contradiction degree — they conflict
        with the most other sections.  This is a graph centrality measure,
        much richer than a simple count from a flat list.

        Returns list of:
          {section, heading, degree, conflicts_with: [...], avg_confidence}
        """
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (s:Section {doc: $doc})-[r:CONTRADICTS]-(other:Section)
                RETURN s.label           AS section,
                       s.heading         AS heading,
                       count(other)      AS degree,
                       collect(DISTINCT other.label) AS conflicts_with,
                       avg(r.confidence) AS avg_confidence
                ORDER BY degree DESC
                LIMIT $n
                """,
                doc=doc_name, n=top_n
            )
            return [dict(r) for r in result]

    def get_transitive_risk(self, section_label: str,
                            doc_name: str) -> list:
        """
        All sections reachable from a given section via contradiction edges,
        with a PROPAGATION SCORE.

        Propagation score decays with each hop:
          score(A→B→C) = conf(A→B) × conf(B→C) × 0.7^(hops-1)
        so a direct High-confidence contradiction scores higher than a
        3-hop Low-confidence chain.

        This is the novel "transitive risk" metric — no existing document
        analysis tool computes this.

        Returns list of:
          {section, hops, propagation_score, path_labels}
        """
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH path = (start:Section {doc: $doc, label: $label})
                  -[:CONTRADICTS*1..4]->(reached:Section {doc: $doc})
                WHERE reached <> start
                WITH reached.label AS section,
                     length(path)  AS hops,
                     [r IN relationships(path) | r.confidence] AS confs,
                     [n IN nodes(path) | n.label] AS path_labels
                WITH section, hops, path_labels,
                     reduce(p = 1.0, c IN confs | p * (c / 100.0)) AS raw_score
                RETURN section,
                       hops,
                       round(raw_score * (0.7 ^ (hops - 1)) * 100) AS propagation_score,
                       path_labels
                ORDER BY propagation_score DESC
                """,
                doc=doc_name, label=section_label
            )
            return [dict(r) for r in result]

    def get_graph_summary(self, doc_name: str) -> dict:
        """Quick stats for the sidebar."""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (s:Section {doc: $doc})
                OPTIONAL MATCH (s)-[r:CONTRADICTS]-()
                RETURN count(DISTINCT s)      AS sections,
                       count(DISTINCT r)      AS edges,
                       avg(r.confidence)      AS avg_confidence
                """,
                doc=doc_name
            )
            row = result.single()
            return {
                "sections":       row["sections"],
                "edges":          row["edges"],
                "avg_confidence": round(row["avg_confidence"] or 0, 1)
            }

    def compare_documents(self, doc1: str, doc2: str) -> list:
        """
        Cross-document query: find contradictions between two stored documents.
        Requires that both documents have been stored and that the checker
        has been run in cross-doc mode (Step 6 of the roadmap).
        """
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (a:Section {doc: $doc1})-[r:CONTRADICTS]->(b:Section {doc: $doc2})
                RETURN a.label  AS section_doc1,
                       b.label  AS section_doc2,
                       r.severity    AS severity,
                       r.confidence  AS confidence,
                       r.type        AS type
                ORDER BY r.confidence DESC
                """,
                doc1=doc1, doc2=doc2
            )
            return [dict(r) for r in result]

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_documents(self) -> list:
        """Return all document names stored in the graph."""
        with self.driver.session() as s:
            result = s.run(
                "MATCH (n:Section) RETURN DISTINCT n.doc AS doc ORDER BY doc"
            )
            return [r["doc"] for r in result]

    def is_available() -> bool:
        """Static check — can be called before constructing the object."""
        return NEO4J_AVAILABLE


def get_graph(uri=None, username=None, password=None) -> ContradictionGraph | None:
    """
    Safe factory: returns a ContradictionGraph if Neo4j is reachable,
    or None if the package isn't installed or the connection fails.
    Use this in app.py so Neo4j being unavailable degrades gracefully.
    """
    if not NEO4J_AVAILABLE:
        return None
    try:
        g = ContradictionGraph(uri, username, password)
        g.driver.verify_connectivity()
        return g
    except Exception:
        return None
