"""
Contradiction Knowledge Graph — Neo4j Integration
===================================================

Builds a graph representation of document contradictions:

  (Section)-[:CONTAINS]->(Statement)
  (Statement)-[:MENTIONS]->(Entity)
  (Statement)-[:CONTRADICTS {score, type, severity}]->(Statement)
  (Section)-[:CONTRADICTS_SECTION {count, max_score}]->(Section)

This enables:
  1. Transitivity detection (A↔B, B↔C → flag A↔C)
  2. Entity-centric clustering (all contradictions about "remote work")
  3. Contradiction centrality (which section is most contradictory?)
  4. Rich graph visualization

Falls back gracefully if Neo4j is not running — the app still works
without it, the graph features just won't be available.

References:
  - Neo4j Python Driver: https://neo4j.com/docs/python-manual/current/
  - Knowledge Graph Construction (ACL 2023)
"""

import os
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

_driver = None

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def get_driver():
    """Get or create Neo4j driver. Returns None if Neo4j is unavailable."""
    global _driver
    if _driver is not None:
        return _driver
    try:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Quick connectivity test
        _driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        return _driver
    except Exception as e:
        logger.warning(f"Neo4j not available ({e}). Graph features disabled.")
        _driver = None
        return None


def is_available() -> bool:
    """Check if Neo4j is reachable."""
    return get_driver() is not None


def close():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION (lightweight, no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

# Common topics in policy/report documents
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
    """Extract topic entities from text using pattern matching."""
    entities = set()
    text_lower = text.lower()
    for pattern, entity_name in ENTITY_PATTERNS:
        if re.search(pattern, text_lower):
            entities.add(entity_name)
    return sorted(entities)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def clear_graph(doc_name: str):
    """Remove all nodes/edges for a specific document."""
    driver = get_driver()
    if not driver:
        return
    with driver.session() as session:
        session.run(
            "MATCH (n {doc_name: $doc}) DETACH DELETE n",
            doc=doc_name
        )


def build_graph(doc_name: str, chunks: list[dict], contradictions: list[dict]):
    """
    Build the full contradiction knowledge graph from chunks and
    detected contradictions.

    Args:
        doc_name: document identifier
        chunks: list of chunk dicts from the pipeline
        contradictions: list of contradiction dicts from run_full_pipeline
    """
    driver = get_driver()
    if not driver:
        return

    clear_graph(doc_name)

    with driver.session() as session:
        # ── Create Document node ────────────────────────────────────
        session.run(
            """
            CREATE (d:Document {name: $name, doc_name: $doc,
                                sections: $n_sections,
                                contradictions: $n_contradictions})
            """,
            name=doc_name, doc=doc_name,
            n_sections=len(chunks),
            n_contradictions=len(contradictions),
        )

        # ── Create Section nodes ────────────────────────────────────
        for chunk in chunks:
            session.run(
                """
                MATCH (d:Document {doc_name: $doc})
                CREATE (s:Section {
                    label: $label,
                    display: $display,
                    heading: $heading,
                    char_count: $chars,
                    doc_name: $doc
                })
                CREATE (d)-[:HAS_SECTION]->(s)
                """,
                doc=doc_name,
                label=chunk["label"],
                display=chunk.get("display", chunk["label"]),
                heading=chunk.get("heading", ""),
                chars=len(chunk["text"]),
            )

            # ── Create Statement nodes + CONTAINS edges ─────────────
            from pipeline.nli_filter import split_into_statements
            statements = split_into_statements(chunk["text"])
            for j, stmt in enumerate(statements):
                stmt_id = f"{chunk['label']}_stmt_{j}"
                session.run(
                    """
                    MATCH (s:Section {label: $label, doc_name: $doc})
                    CREATE (st:Statement {
                        id: $stmt_id,
                        text: $text,
                        section_label: $label,
                        doc_name: $doc
                    })
                    CREATE (s)-[:CONTAINS]->(st)
                    """,
                    doc=doc_name,
                    label=chunk["label"],
                    stmt_id=stmt_id,
                    text=stmt,
                )

                # ── Create Entity nodes + MENTIONS edges ────────────
                entities = extract_entities(stmt)
                for entity in entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name, doc_name: $doc})
                        WITH e
                        MATCH (st:Statement {id: $stmt_id, doc_name: $doc})
                        MERGE (st)-[:MENTIONS]->(e)
                        """,
                        doc=doc_name, name=entity, stmt_id=stmt_id,
                    )

        # ── Create CONTRADICTS edges ────────────────────────────────
        for c in contradictions:
            analysis = c.get("analysis", "")

            # Extract the two statements from the analysis
            stmt_a_match = re.search(r'Statement A[^"]*"([^"]+)"', analysis)
            stmt_b_match = re.search(r'Statement B[^"]*"([^"]+)"', analysis)

            if stmt_a_match and stmt_b_match:
                stmt_a_text = stmt_a_match.group(1)[:200]
                stmt_b_text = stmt_b_match.group(1)[:200]

                # Link statements via text matching
                session.run(
                    """
                    MATCH (a:Statement {doc_name: $doc})
                    WHERE a.text CONTAINS $stmt_a
                    WITH a LIMIT 1
                    MATCH (b:Statement {doc_name: $doc})
                    WHERE b.text CONTAINS $stmt_b
                    WITH a, b LIMIT 1
                    CREATE (a)-[:CONTRADICTS {
                        score: $score,
                        type: $type,
                        severity: $severity,
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

            # Also create section-level contradiction edge
            target_label = c.get("target", "")
            for related in c.get("related_sections", []):
                related_clean = related.replace(" (same section)", "")
                session.run(
                    """
                    MATCH (a:Section {label: $label_a, doc_name: $doc})
                    MATCH (b:Section {label: $label_b, doc_name: $doc})
                    MERGE (a)-[r:CONTRADICTS_SECTION]->(b)
                    ON CREATE SET r.count = 1, r.max_score = $score,
                                  r.types = [$type]
                    ON MATCH SET r.count = r.count + 1,
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

    logger.info(f"Built knowledge graph for '{doc_name}': "
                f"{len(chunks)} sections, {len(contradictions)} contradictions")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def get_transitive_contradictions(doc_name: str) -> list[dict]:
    """
    Find transitive contradictions: if Statement A contradicts B,
    and B contradicts C, then A and C may also be inconsistent.
    This is a novel multi-hop reasoning feature.
    """
    driver = get_driver()
    if not driver:
        return []

    with driver.session() as session:
        result = session.run(
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
            doc=doc_name
        )
        return [dict(record) for record in result]


def get_entity_contradiction_clusters(doc_name: str) -> list[dict]:
    """
    Group contradictions by the entities they mention.
    Shows which topics have the most contradictory statements.
    """
    driver = get_driver()
    if not driver:
        return []

    with driver.session() as session:
        result = session.run(
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
            doc=doc_name
        )
        return [dict(record) for record in result]


def get_most_contradictory_sections(doc_name: str) -> list[dict]:
    """
    Rank sections by contradiction centrality —
    how many contradiction edges touch each section.
    """
    driver = get_driver()
    if not driver:
        return []

    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Section {doc_name: $doc})
            OPTIONAL MATCH (s)-[:CONTAINS]->(st:Statement)-[r:CONTRADICTS]-()
            WITH s.label AS section, s.display AS display,
                 count(r) AS contradiction_edges
            RETURN section, display, contradiction_edges
            ORDER BY contradiction_edges DESC
            """,
            doc=doc_name
        )
        return [dict(record) for record in result]


def get_graph_stats(doc_name: str) -> dict:
    """Get summary statistics for the knowledge graph."""
    driver = get_driver()
    if not driver:
        return {}

    with driver.session() as session:
        result = session.run(
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
            doc=doc_name
        )
        record = result.single()
        return dict(record) if record else {}
