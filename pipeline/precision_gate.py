"""
Precision Gate — single entry point wiring:

    role_classifier  ➜  negation_gate  ➜  claim_extractor  ➜  arithmetic_reconciler

Call ``accept(sent_a, sent_b, universe=None)`` — it returns a
``Decision`` with ``keep=True/False`` and a ``reason`` so the caller
can log why a pair was accepted or rejected.

The order matters:

    1. Role compatibility     (kills narrative / recommendation / caption)
    2. Negation-gate signals  (kills adjacency-only "direct" pairs)
    3. Claim extraction       (kills different-metric / different-entity pairs)
    4. Arithmetic reconciler  (kills sum-of-parts, share-of-whole, rounding)

Anything surviving all four gates is genuinely *candidate-worthy* — it
then goes to the NLI scorer (already in nli_filter.py) and, if still
above threshold, to the LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pipeline.role_classifier import classify, compatible
from pipeline.negation_gate import gate
from pipeline.claim_extractor import (
    extract_claims,
    claims_comparable,
    claims_contradict,
)
from pipeline.arithmetic_reconciler import reconcile_pair


@dataclass
class Decision:
    keep: bool
    reason: str
    stage: str                       # which gate made the call
    signals: list[str]               # supporting evidence
    detail: dict


def accept(
    sent_a: str,
    sent_b: str,
    universe: Optional[list[float]] = None,
    numeric_rel_tol: float = 0.03,
) -> Decision:
    """
    Run the full gating chain. Returns keep=False as soon as any gate
    rejects the pair; keep=True only if all gates pass OR if the claim
    extractor finds a real numeric contradiction.
    """
    # ── Stage 1: role compatibility ─────────────────────────────────
    ra = classify(sent_a).role
    rb = classify(sent_b).role
    if not compatible(ra, rb):
        return Decision(
            keep=False,
            reason=f"role_mismatch ({ra} ↔ {rb})",
            stage="role",
            signals=[],
            detail={"role_a": ra, "role_b": rb},
        )

    # ── Stage 2: negation / antonym / modal / numeric-delta gate ────
    g = gate(sent_a, sent_b)

    # ── Stage 3: claim extraction ───────────────────────────────────
    claims_a = extract_claims(sent_a)
    claims_b = extract_claims(sent_b)

    # Fast-path: a real numeric contradiction between comparable claims.
    # These we KEEP even before arithmetic reconciliation — then the
    # reconciler gets a second chance to clear them.
    for ca in claims_a:
        for cb in claims_b:
            if claims_contradict(ca, cb, rel_tol=numeric_rel_tol):
                # Try to reconcile via universe (sum-of-parts, etc.)
                if universe:
                    r = reconcile_pair(ca.value, cb.value, universe=universe)
                    if r.consistent:
                        return Decision(
                            keep=False,
                            reason=f"arithmetic_reconciled: {r.relation}",
                            stage="reconciler",
                            signals=list(g.signals),
                            detail={"relation": r.relation, "detail": r.detail},
                        )
                return Decision(
                    keep=True,
                    reason="claim_contradiction",
                    stage="claim",
                    signals=list(g.signals),
                    detail={
                        "a": ca.as_dict(),
                        "b": cb.as_dict(),
                        "gap": abs(ca.value - cb.value) / max(abs(ca.value), abs(cb.value), 1e-9),
                    },
                )

    # If claims exist on both sides but NONE are comparable, the pair
    # is arithmetically unrelated → drop.
    if claims_a and claims_b:
        any_comparable = any(
            claims_comparable(ca, cb) for ca in claims_a for cb in claims_b
        )
        if not any_comparable:
            return Decision(
                keep=False,
                reason="no_comparable_claims (different metric/entity/unit/population)",
                stage="claim",
                signals=list(g.signals),
                detail={
                    "claims_a": [c.as_dict() for c in claims_a],
                    "claims_b": [c.as_dict() for c in claims_b],
                },
            )

    # ── Stage 4: negation-gate result (for DIRECT/antonym/modal) ────
    if not g.passed:
        return Decision(
            keep=False,
            reason="no_polarity_signal",
            stage="negation_gate",
            signals=[],
            detail=dict(g.detail),
        )

    # All gates passed. Hand to NLI / LLM downstream.
    return Decision(
        keep=True,
        reason="gates_passed",
        stage="all_passed",
        signals=list(g.signals),
        detail=dict(g.detail),
    )


# ─────────────────────────────────────────────────────────────────────
# Smoke test — exercise the 16 false positives + real contradictions
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # universe = the canonical numbers in the Cloud report
    universe = [
        395.0, 490.0, 591.0, 712.0, 843.0, 992.0, 1157.0, 1338.0, 1572.0, 1820.0,
        20.9, 16.6, 18.5,
        42.0, 26.0, 23.0, 5.0, 4.0,
        31.0, 25.0, 12.0, 3.0, 2.0, 23.0, 68.0,
        41.0, 32.0, 20.0, 7.0,
        94.0, 73.0, 82.0, 61.0, 58.0, 29.0,
        32.0, 18, 24,
        2.1, 138, 5.4,
    ]

    fps = [
        # #1 Section 2 intra
        ("Executive Summary Global cloud-computing spend reached $843 billion in 2025, a compound annual growth rate of 20.9% since 2021.",
         "The cloud-computing market has moved decisively from a growth story into an operating-scale story."),
        # #2 Section 4 vs 9 (figure captions)
        ("Figure 2 — Regional share of the 2025 cloud market.",
         "Figure 6 — Forecast regional CAGR, 2025-2030."),
        # #3 Section 6 intra
        ("The SaaS share reflects accumulated migration from on-premise packaged software.",
         "IaaS maintains its share through the combination of lift-and-shift migration of legacy workloads and the GPU-heavy compute demands of model training."),
        # #4 Section 7 intra
        ("Our 1,200-enterprise survey, conducted across 24 countries, finds that 94% of respondents run at least one workload on a public-cloud provider.",
         "Edge-cloud deployments, where compute is pushed closer to users or devices, are still emerging: 29% of enterprises report an active pilot, and commercial scale remains selective."),
        # #5 Section 7 vs 2
        ("Edge-cloud deployments are still emerging: 29% of enterprises report an active pilot.",
         "The cloud-computing market has moved decisively from a growth story into an operating-scale story."),
        # #6 Section 8 intra
        ("Enterprises that migrated in disciplined waves report payback periods inside twenty-four months and continuing opex reductions after the transition.",
         "Enterprises that lifted-and-shifted without re-architecting frequently observe flat or higher total costs in years two and three."),
        # #7 Section 8 vs 7 (different denominators)
        ("Professional services — both hyperscaler-led and partner-delivered — are 26%, primarily driven by migration and re-platforming work.",
         "58% of respondents report that a majority of new application development is cloud-native."),
        # #8 Section 8 vs 10 (claim vs recommendation)
        ("Enterprises that lifted-and-shifted without re-architecting frequently observe flat or higher total costs in years two and three.",
         "- Negotiate committed-spend and reserved-capacity terms in parallel rather than sequentially."),
        # #9 Section 10 intra (claim vs recommendation)
        ("Security and compliance remain the most-cited concern, named by 58% of respondents, with cost predictability at 49%.",
         "- Negotiate committed-spend and reserved-capacity terms in parallel rather than sequentially."),
        # #10 captions
        ("Figure 2 — Regional share of the 2025 cloud market.",
         "Figure 3 — Hyperscaler market share, 2025."),
        # #11 AWS 31% vs top-3 68% (sum-of-parts)
        ("Amazon Web Services remains the single largest individual provider at 31%, followed by Microsoft Azure at 25% and Google Cloud at 12%.",
         "First, hyperscaler concentration is deepening: Amazon Web Services, Microsoft Azure, and Google Cloud collectively hold 68% of global cloud spend."),
        # #12 Section 11 intra (two data-source bullets)
        ("- Hyperscaler financial disclosures: public filings and investor-day material from the seven largest cloud providers, normalized to a calendar-year basis.",
         "- Public-sector procurement data: published cloud-services contracts from national and supranational procurement bodies across the Atlas-coverage universe."),
        # #13 Section 7 vs 11 ("past 24 months" vs "24 countries")
        ("Beyond basic penetration, the 2025 survey highlights three adoption vectors that have matured in the past twenty-four months.",
         "Primary survey: 1,200 enterprise respondents across 24 countries, fielded in Q1 2026."),
        # #14 Section 9 vs 2
        ("Middle East & Africa and Latin America will grow fastest on a percentage basis but remain single-digit share of global spend through 2030 in our base case.",
         "Executive Summary Global cloud-computing spend reached $843 billion in 2025, a compound annual growth rate of 20.9% since 2021."),
        # #15 Section 7 vs 6
        ("Edge-cloud deployments are still emerging: 29% of enterprises report an active pilot.",
         "Horizontal suites (productivity, communications, collaboration) are effectively fully cloud-delivered for new deployments."),
        # #16 Section 3 vs 2
        ("The global cloud-computing market reached $843 billion in 2025, expanding from $395 billion in 2021.",
         "The cloud-computing market has moved decisively from a growth story into an operating-scale story."),
    ]

    tps = [
        # Real contradictions the gate MUST keep
        ("All employees must use VPN when working remotely.",
         "Employees working remotely do not need to use VPN."),
        ("Revenue rose to $412 million in fiscal 2024.",
         "Revenue fell to $389 million in fiscal 2024."),
        ("Maximum daily transaction limit is $10,000.",
         "Maximum daily transaction limit is $25,000."),
        # Real numeric contradictions
        ("AWS holds 31% of global cloud spend in 2025.",
         "AWS holds 42% of global cloud spend in 2025."),
    ]

    print("=== 16 known false positives (should ALL be DROP) ===")
    fp_kept = 0
    for i, (a, b) in enumerate(fps, 1):
        d = accept(a, b, universe=universe)
        marker = "KEEP" if d.keep else "drop"
        if d.keep:
            fp_kept += 1
        print(f"  #{i:2d} [{marker}] stage={d.stage:12s}  reason={d.reason}")

    print(f"\n=== 4 true contradictions (should ALL be KEEP) ===")
    tp_dropped = 0
    for i, (a, b) in enumerate(tps, 1):
        d = accept(a, b, universe=universe)
        marker = "KEEP" if d.keep else "DROP"
        if not d.keep:
            tp_dropped += 1
        print(f"  #{i:2d} [{marker}] stage={d.stage:12s}  reason={d.reason}")

    print(f"\n=== SUMMARY ===")
    print(f"  False positives kept (want 0):     {fp_kept} / {len(fps)}")
    print(f"  True contradictions dropped (want 0): {tp_dropped} / {len(tps)}")
